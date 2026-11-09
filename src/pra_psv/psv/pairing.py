"""Offline intervention pairs; contexts are independently copied before execution."""
import copy
from .core import paired_targets


def collect_pair(context, rollout, *, teacher=None):
    """rollout(context, force_review, teacher) returns outcome and causal pack.

    The integration must seed both continuations identically, use the same
    frozen future policy, and derive outcomes from actually executed actions.
    No test context may be passed to this training interface.
    """
    if context["split"] not in ("train", "validation_select"):
        raise ValueError("Only training/selection contexts may generate training pairs")
    keep, pack = rollout(copy.deepcopy(context), False, teacher)
    reviewed, _ = rollout(copy.deepcopy(context), True, teacher)
    return dict(id=context["id"], group=context["group"], split=context["split"],
                pack=pack, targets=paired_targets(keep, reviewed)), {"keep": keep, "review": reviewed}

