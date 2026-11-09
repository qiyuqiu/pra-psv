"""Balanced selective review: frozen margin or supported learned benefit.

Uncertainty is not attack evidence. Never extrapolate a learner beyond its
training budget support. Skipping review preserves the PRA decision; it does
not authorize an action. No actual-review cap is imposed.
"""
from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Decision:
    review: bool
    reasons: tuple[str, ...]

def decide(*, tool, observation_eligible, pra_score, prediction, reviews_used,
           read_only_tools, margin_cutoff, pra_threshold=.89):
    if (type(reviews_used) is not int or reviews_used < 0
        or not math.isfinite(pra_score) or not 0 <= pra_score <= 1
        or not math.isfinite(margin_cutoff) or margin_cutoff < 0
        or not math.isfinite(pra_threshold) or not 0 <= pra_threshold <= 1):
        raise ValueError('Invalid routing inputs')
    if not observation_eligible:
        return Decision(False, ('no_eligible_observation',))
    reasons = []
    diagnostics = []
    if abs(pra_score-pra_threshold) <= margin_cutoff:
        reasons.append('frozen_margin_trigger')
    if reviews_used >= 2:
        diagnostics.append('margin_only_outside_trained_budget_support')
    elif prediction is None:
        diagnostics.append('margin_fallback_missing_prediction')
    else:
        required = ('abstain', 'unsupported_branch', 'out_of_support', 'selected')
        if any(type(prediction.get(k)) is not bool for k in required):
            raise ValueError('Malformed router prediction; never treat corruption as uncertainty')
        if any(prediction[k] for k in required[:3]):
            diagnostics.append('margin_fallback_uncertain_or_unsupported')
        elif prediction['selected']:
            reasons.append('supported_learned_trigger')
        else:
            diagnostics.append('supported_learner_not_selected')
    return Decision(bool(reasons), tuple(reasons+diagnostics))
