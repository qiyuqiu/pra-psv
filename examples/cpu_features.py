"""Run from an editable installation; no backbone, GPU or external service."""
from pra_psv.assets import pairs
from pra_psv.psv.routing import PSVRouter

row = pairs(1)[0]
decision, prediction = PSVRouter.from_release().select_review(
    tool='example_action', observation_eligible=True,
    pra_score=row['pack']['numeric'][0], reviews_used=0, pack=row['pack'])
print({'review': decision.review, 'reasons': decision.reasons, 'prediction': prediction})
