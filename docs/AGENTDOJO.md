# AgentDojo integration

## External dependencies

Obtain [AgentDojo](https://github.com/ethz-spylab/agentdojo) separately. The reference benchmark version is `v1.2.2`; the inspected upstream runner revision is `089ed468cf3ed0322acc66b0211f26d9d90dbf60`. Install its dependencies in your own environment and preserve the benchmark license.

Obtain the authors' [AttriGuard artifact](https://doi.org/10.5281/zenodo.20308739) and its `AttriGuard.py` separately. The reference file SHA-256 is `6d28e2208efbd521bf3f2e90c553e57b11c786e65564eababacb8cdf4f8050d8`. The wrapper calls the artifact's `AttriGuardExecutionLoop`, `_attenuate_observation`, `_fuzzy_survive`, `_normalize_messages`, `_blocked_tool_result`, and `EXTERNAL_OUTPUT_TOOLS`; incompatible revisions must not be silently substituted. This is a component adaptation, not the unmodified AttriGuard evaluation pipeline.

## Construct the gate

The following is integration code: `author`, `actor`, `attenuation_llm` and `judge_llm` are explicitly loaded/configured upstream components, rather than names supplied by this package.

```python
from pra_psv.pra.encoding import FrozenPrefixEncoder
from pra_psv.pra.inference import PRAMonitor
from pra_psv.psv.encoding import StateEncoder
from pra_psv.psv.routing import PSVRouter
from pra_psv.integrations.agentdojo import AgentDojoGate, AttriGuardReviewer

encoder = FrozenPrefixEncoder('models/Qwen2.5-7B-Instruct', max_length=512)
pra = PRAMonitor.from_release(encoder=encoder)
reviewer = AttriGuardReviewer(author, actor, attenuation_llm, judge_llm)
state_encoder = StateEncoder(encoder, author.EXTERNAL_OUTPUT_TOOLS)
gate = AgentDojoGate(pra, router=PSVRouter.from_release(),
                    state_encoder=state_encoder, reviewer=reviewer)
# PRA alone: gate = AgentDojoGate(pra)

# After actor.query has appended its proposed assistant tool calls:
query, runtime, env, messages, extra = gate.query(query, runtime, env, messages, {})
```

Construct the monitor cache, reviewer cache and gate afresh for each episode. Sharing the immutable encoder and model weights is possible, but do not carry episode history, review counters or caches into the next episode when measuring the reference fresh-cache protocol.

Use the same actor for real and shadow generation. Attenuation and fuzzy judging are actual model calls. The benchmark runtime's functions and environments remain under AgentDojo's control. The caller must terminate on the actor's final answer or the 15-round limit, preserve all raw messages, and adjudicate from executed calls rather than merely proposed calls.

Reference generations use temperature 0 and 512 output tokens with a 32,768-token context. Reference feature forwards use BF16 encoder inference. Do not use outputs from a different actor or generation budget to claim an unchanged paper evaluation.

## Offline paired collection

`psv.pairing.collect_pair` takes a causal context and a caller-supplied rollout callback. Each arm receives an independent deep copy. The callback resets the random seed, executes retain-PRA or force-review at the candidate, then uses the same frozen future teacher policy. It returns a benchmark-adjudicated outcome and the candidate's causal feature pack. Keep all raw outcomes independently; unknown/ambiguous outcomes produce masked targets.

The three targets are safe-completion gain, attack prevention and utility gain. The learned preference is safe gain + 2 * attack prevention; utility gain is an auxiliary output. Round 1 requires new paired continuations under the frozen round-0 teacher. No test IDs, attack strategy labels, oracle paths or future outcomes may enter the online feature pack.
