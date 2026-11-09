# Reproducibility boundaries

## What the release verifies

The policy, four-view feature functions, PRA head/objective and benefit-ensemble implementation were extracted from the frozen reference implementation. Release checks compare their outputs against that implementation on saved inputs. Removing unrelated historical helpers does not change the retained computations.

The four reference weight files were reserialized after replacing personal path metadata. Every tensor, tensor shape and scalar parameter used for inference remains equal to its source. Original and release hashes are listed in `artifacts/checkpoints.json`. Training pair fields used by fitting retain their values and order; raw-path and deployment provenance fields are omitted. PRA adaptation retains only the PRA feature tensors and required labels/groups, omitting unrelated model tensors.

CPU equivalence and import tests establish packaging compatibility. They do not establish full end-to-end equivalence under changed model-serving software or prove any improvement in attack success. The portable AgentDojo adapter requires a fresh native integration check before collecting new benchmark results.

## What requires external inputs

- Frozen backbone weights and tokenizer, pinned to the stated revision.
- Original DecomposedHarm data, inner split and teacher records for DH retraining.
- AgentDojo environments, attack inputs and official evaluators for new execution.
- The released AttriGuard implementation and actual attenuation/judge pipelines.
- Additional backbone-specific feature caches, checkpoints and full raw outcome ledgers to reproduce all paper tables.

The reference training caches are Qwen7B only. Fitting independent cached rounds is not full closed-loop retraining. Runs on different devices/software can select different checkpoints, even with equal seeds. Refitting must write new output directories and must not overwrite the released weights.

## Dataset and interpretation limitations

The AD action labels combine benchmark-confirmed attack-goal transitions and weak clean/reference-action evidence. They are not human authorization certificates. Selector targets are trajectory-level interventions and are not per-action safety labels. Limited preference support and ensemble disagreement constrain learned selection; a fixed-margin fallback remains active.

Task-level train/selection/audit groups are separate. Three held-out attack strategies are not used for training. Historical benchmark development and previously inspected test results limit claims of independent generalization; this artifact does not erase that history or present new blind evaluation.

Experimental result reports and numerical performance comparisons are excluded. Stored training diagnostics are removed from distributed PSV checkpoints; inference weights, preprocessing and decision thresholds are preserved.

## No implicit execution

Importing or installing this package does not launch remote jobs, fit models, make API requests, start schedulers or resume old experiments. `verify` and `demo` are CPU-only. The training commands act only when explicitly invoked. Text encoding explicitly loads the supplied backbone.
