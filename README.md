# PRA and PSV

Research artifact for **Prefix Risk Adapter (PRA)** and **PRA-guided Selective Verification (PSV)**.

PRA predicts cumulative risk from a frozen language model's prefix representations. PSV selectively invokes action verification when a fixed PRA margin or a supported learned benefit estimate requests review. A review may block an unsafe action or release an action that PRA would otherwise block.

This repository contains the two methods, their training objectives, causal feature construction, CPU inference interfaces, a portable AgentDojo gate, and reference Qwen2.5-7B checkpoints and cached training inputs. Historical research methods, server administration scripts, credentials, private experiment logs, and the author manuscript are excluded.

## 1. Release scope

| Component | Included |
|---|---|
| PRA architecture and DH objective | Frozen encoder interface, two-layer risk head, task-normalized loss, teacher distillation, benign maximum-prefix penalty |
| PRA training | DH data loader and trainer; AD weighted adaptation on the released 165/57/61 cached states |
| PSV | Four text views, 44 numeric features, train-only PCA, three-member benefit ensemble, support and disagreement checks, balanced review policy |
| Reference artifacts | Qwen7B PRA-DH/PRA-AD heads, both PSV fitting-round checkpoints, both cached pair rounds |
| AgentDojo integration | Pre-execution gate and adapter to the authors' AttriGuard attenuation/survival implementation |

The supplied pretrained selector is the **original balanced Qwen7B reference**, with source checkpoint SHA-256 `17a7bdd2fd6643a0fdc60905109309421ce20ffda20f50f632638b49016abb3e`. The PRA-AD source checkpoint is `0deee2b76649bfcc30e93c9abcaf8367f9264126dd7d3783b4dca45d20d2f977`.

The paper evaluates additional backbones and an API actor. Their weights and complete evaluation traces are **not bundled** here. The supplied Qwen7B artifacts do not recreate every row of the paper. Later experimental retraining and review-denial-memory repairs are excluded from this reference release. See [reproducibility details](docs/REPRODUCIBILITY.md).

## 2. Installation

Python 3.10 or newer is required. CPU verification and cached-feature inference need NumPy and PyTorch; they do not download a backbone or contact an API.

From the extracted repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pra_psv verify
python -m pra_psv demo
python -m pytest -q
```

For text encoding with an explicitly supplied Hugging Face model:

```bash
python -m pip install -e '.[encoder]'
```

This release was checked with Python 3.11.5, PyTorch 2.14.0 and NumPy 2.4.6. The dependency ranges also permit other versions; byte-identical retraining across environments is not guaranteed. Install a PyTorch build suitable for your device if performing new encoder inference.

Use an editable install, as above, so that the package can locate the repository's `artifacts/`, `data/` and `configs/` directories. For a copied package or another working layout, set `PRA_PSV_ROOT` to the extracted repository root.

## 3. Repository layout

```text
src/pra_psv/
  pra/
    model.py              PRA head and DH training objective
    encoding.py           frozen final-token encoder
    inference.py          feature and visible-prefix risk scoring
    train_ad.py           PRA-only AD adaptation
    dh/                   DH schema, loader, metrics and fitting code
  psv/
    prefix.py             visible-prefix serialization and base features
    features.py           four causal text views and extra numeric features
    encoding.py           construct a 4 x 3584 + 44 reference feature pack
    model.py              train-only preprocessing and benefit ensemble
    policy.py             balanced margin/learned selection policy
    routing.py            pretrained selector API
    pairing.py            independent offline intervention-pair interface
  integrations/
    agentdojo.py          PRA/PSV pre-execution gate and reviewer adapter
  cli.py                  explicit check, demo and training commands
artifacts/                anonymized reference checkpoints and hashes
data/pra_ad/              PRA-only action-state cache
data/psv/                 two rounds of cached training/selection pairs
configs/                  reference working points and hyperparameters
docs/                     integration, data and reproduction details
examples/                 executable CPU feature demonstration
tests/                    policy, leakage, checkpoint and integration controls
checksums.json            release-file integrity manifest
```

Each data shard is below 4 MiB. Checkpoints are gzip-compressed below 8 MiB per file for anonymous repository browsing/download compatibility. They contain actual weights, rather than Git LFS pointers. `artifacts/checkpoints.json` records both source and released file hashes. Anonymization changes serialization metadata and thus the file hash; weight tensors and numerical training inputs are preserved.

## 4. PRA

PRA feeds a frozen final-token hidden vector and normalized prefix length into a LayerNorm and two 512-unit hidden layers with GELU activations. It applies a sigmoid to obtain the risk score. Only the small head is trained; inference still requires frozen-backbone computation when features are not cached.

The DH loss includes prefix classification, masked teacher distillation, an any-prefix task term, and a benign maximum-prefix penalty. Harmful training prefixes stop at the annotated harm step. Labels, future steps and the original harmful goal are not encoder inputs. AD adaptation uses weighted action-level binary cross-entropy.

To score the released AD features on CPU:

```python
import gzip
import io
import torch
from pra_psv.assets import release_root
from pra_psv.pra.inference import PRAMonitor

with gzip.open(release_root() / 'data/pra_ad/features.pt.gz', 'rb') as stream:
    cache = torch.load(io.BytesIO(stream.read()), map_location='cpu', weights_only=True)
monitor = PRAMonitor.from_release(benchmark='ad')
scores = monitor.score_features(cache['pra'], cache['counts'])
blocked = scores >= monitor.threshold   # reference AD threshold: 0.89
```

For new text, instantiate `FrozenPrefixEncoder` with a local model snapshot and pass it to `PRAMonitor.from_release(encoder=encoder)`. The reference snapshot is `Qwen/Qwen2.5-7B-Instruct`, revision `a09a35458c702b33eeacc393d103063234e8bc28`; its hidden width is 3584. The feature limit is 512 tokens. The AD monitor separately rejects visible prefixes above 8192 tokens and skips risk scoring before two proposed actions, matching the reference gate.

Do not substitute a different encoder revision, hidden width, tokenizer or truncation rule while treating a checkpoint as unchanged.

## 5. PSV

The selector uses request, action, recent external evidence and joint views. Each view is encoded with the same frozen Qwen7B encoder. Train-only PCA reduces the normalized vectors to 32 dimensions. Four projected vectors, two elementwise interaction vectors and 44 numeric values form a 236-dimensional input. An ensemble of three small networks predicts safe-completion gain, attack prevention and utility gain, conditioned on whether PRA blocks the candidate.

```python
from pra_psv.assets import pairs
from pra_psv.psv.routing import PSVRouter

row = pairs(1)[0]  # demonstration on a cached TRAINING feature pack
router = PSVRouter.from_release()
decision, prediction = router.select_review(
    tool='example_action',
    observation_eligible=True,
    pra_score=row['pack']['numeric'][0],
    reviews_used=0,
    pack=row['pack'],
)
print(decision.review, decision.reasons)
```

`decision.review` requests verification; it is not an allow decision. If review is skipped, use PRA's allow/block result. If review is selected, the actual attenuation/survival reviewer supplies the final decision.

The fixed margin is `abs(score - 0.89) <= 0.2140339875221252`. Learned selection additionally requires positive benefit, validation-set cutoffs, supported training branches and in-support features. Ensemble disagreement is not a calibrated safety probability. The reference learned selector is used only while fewer than two reviews have already occurred. **Margin-triggered reviews remain available after that point; two is not a cap on total reviews.** A tool category alone never triggers review.

## 6. Training

All commands require a new output directory. Nothing starts automatically when the package is imported or installed.

### PRA adaptation on released AD features

```bash
python -m pra_psv train-pra-ad --device cpu --out runs/pra-ad-refit
```

This uses the frozen 165 training, 57 selection and 61 audit action states, seeds 17/29/43, 20 epochs, AdamW learning rate `1e-4`, batch size 32 and gradient clipping 1. Validation BCE selects the checkpoint; audit states do not select it. The deployment threshold remains 0.89. `--device cuda:0` explicitly requests GPU head fitting in a separately prepared environment. Feature extraction and actor rollouts are not repeated.

### PSV cached pair refitting

```bash
python -m pra_psv train-psv --round 0 --out runs/psv-round0-refit
python -m pra_psv train-psv --round 1 --out runs/psv-round1-refit
```

Round 0 has 241 records and round 1 has 227. Indeterminate pair targets remain `null` and are masked. Each fit uses only train/selection groups, three seeds, 60 epochs, AdamW learning rate 0.003, train-only preprocessing and validation-only member selection.

These commands refit each round's **existing cached pairs** independently. A newly fitted round-0 model is not substituted into the historical round-1 collection. Full closed-loop retraining requires generating new round-1 intervention pairs under the new frozen round-0 teacher. `psv/pairing.py` supplies the pair interface; the benchmark-specific causal-state reconstruction and rollout evaluator must be supplied by the integration. Unknown prior pairs must remain unknown and must not be retried to obtain preferred labels.

### PRA DH training

The raw DecomposedHarm release, original inner training split, teacher-score records and base-model weights must be obtained separately. The loader expects the upstream repository root, containing `data/combined/`, `data/decomposed_queries/`, `data/image_tasks/` and `data/agent_tasks/`.

```bash
python -m pra_psv train-pra-dh \
  --data-dir external/monitoring_decomposition_attack \
  --split external/dh_inner_split.json \
  --teacher external/teacher_inner.jsonl \
  --encoder models/Qwen2.5-7B-Instruct \
  --out runs/pra-dh-new
```

The split JSON must contain `inner_train_ids` with 2,595 unique qualified task IDs. Validation uses the 264 upstream combined validation tasks. Teacher JSONL records contain `task_id`, `prefix_length`, `teacher_probability` and `error`. Missing/erroneous teacher scores are masked. The command explicitly performs encoder inference, then ten epochs of head fitting with seed 11 and validation F1 selection. Its output is a new run; software/device differences may change its weights.

## 7. AgentDojo execution

The optional adapter targets AgentDojo v1.2.2 and the official AttriGuard implementation. These upstream projects and their data/licenses are obtained separately. See [integration instructions](docs/AGENTDOJO.md).

The caller provides actor, attenuation and judge pipelines, the local encoder, and the official reviewer module. Create fresh monitor/selector-state/reviewer/gate instances for every episode. Replace the pipeline's tool-execution element with `AgentDojoGate`; do not also execute the original candidate batch. The caller runs the actor loop for at most 15 rounds and records actual executions, raw responses, costs and unknown outcomes.

The adapter preserves current-batch visibility, handles invalid tool names before execution, reuses one shadow batch within an actor turn, and follows PRA or the reviewer before invoking a tool. It does not implement API transport, model serving, resource scheduling, benchmark task reconstruction or a one-command reproduction of the entire paper. The portable integration itself has not been re-evaluated over 1,770 GPU episodes as part of packaging.

## 8. Evaluation interfaces

Experimental result tables, aggregate reports and per-episode evaluation outputs are not included in this release.

DH task detection is timely only if the first alarm occurs by the annotated harm step; alarms on a single-subtask prefix do not count. Thresholds are selected on validation, not test data. `pra/dh/metrics.py` implements task F1, FPR and the low-FPR working point.

AD evaluation uses attack success `A`, user completion `U`, and safe completion `U * (1 - A)`. ASR uses attacked episodes, Safe uses all retained episodes, and clean utility uses only clean episodes. Any new evaluation must document its cohort and distinguish observed outcomes from conservative assignments for indeterminate episodes.

Costs must be measured from the corresponding execution records. Shared hardware, API waits and feature caching affect episode time; cached CPU inference is not an end-to-end speed measurement.

## 9. Integrity, anonymity and attribution

Run `python -m pra_psv verify` to check release hashes, all four checkpoints, cached feature inference and disjoint train/selection groups. Run `python -m pytest -q` for policy and interface controls.

The release omits names, affiliations, author email addresses, original Git history, personal filesystem paths, host addresses, credentials and operational logs. Binary checkpoints were inspected and personal encoder-path metadata was replaced without changing tensors. Upstream benchmark/tool names and public third-party citations are retained for attribution. The manuscript is not part of the repository. A personal GitHub hosting account and its commit metadata remain visible, so a link to that account is not an anonymous review link.

No open-source license has been assigned to the original research implementation in this research snapshot. Third-party materials remain subject to their respective licenses; see [third-party sources](THIRD_PARTY_NOTICES.md). Please cite the associated paper after review; identifying citation metadata is intentionally omitted here.
