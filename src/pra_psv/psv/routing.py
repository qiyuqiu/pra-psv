"""Select review using the frozen balanced PSV policy."""
import json
from .model import Router
from .policy import decide
from ..assets import checkpoint, release_root


class PSVRouter:
    def __init__(self, router, margin_cutoff=.2140339875221252, pra_threshold=.89):
        self.router = router
        self.margin_cutoff = margin_cutoff
        self.pra_threshold = pra_threshold

    @classmethod
    def from_release(cls):
        package = checkpoint("psv_round1")
        if package["format"] != "psv_action_router_ensemble_v1":
            raise ValueError("Unsupported router format")
        config = json.loads((release_root() / "configs/psv.json").read_text())
        return cls(Router(package), config["margin_cutoff"], config["pra_threshold"])

    def select_review(self, *, tool, observation_eligible, pra_score, reviews_used, pack=None):
        prediction = None
        if observation_eligible and reviews_used < 2 and self.router is not None and pack is not None:
            prediction = self.router.predict(pack)
        return decide(tool=tool, observation_eligible=observation_eligible,
                      pra_score=pra_score, prediction=prediction, reviews_used=reviews_used,
                      read_only_tools=frozenset(), margin_cutoff=self.margin_cutoff,
                      pra_threshold=self.pra_threshold), prediction

