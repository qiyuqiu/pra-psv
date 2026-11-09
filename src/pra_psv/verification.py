"""Release integrity and CPU compatibility; does not run benchmark episodes."""
import gzip
import io
import json
from .assets import checkpoint, pairs, release_root, sha256


def verify():
    import torch
    from .pra.inference import PRAMonitor
    from .psv.routing import PSVRouter
    root = release_root()
    manifest = json.loads((root / "checksums.json").read_text())
    for path, expected in manifest.items():
        if sha256(root / path) != expected:
            raise ValueError("Release checksum mismatch: " + path)
    for name in ("pra_dh", "pra_ad", "psv_round0", "psv_round1"):
        checkpoint(name)
    counts = {}
    router = PSVRouter.from_release()
    for ri in (0, 1):
        rows = pairs(ri)
        train = {r["group"] for r in rows if r["split"] == "train"}
        select = {r["group"] for r in rows if r["split"] == "validation_select"}
        if train & select or any(r["split"] not in ("train", "validation_select") for r in rows):
            raise ValueError("Invalid pair partitions")
        for row in rows:
            pack = row["pack"]
            if len(pack["numeric"]) != 44 or len(pack["views"]) != 4 or any(len(v) != 3584 for v in pack["views"]):
                raise ValueError("Unexpected reference feature shape")
        for row in rows[:8]:
            router.router.predict(row["pack"])
        counts[str(ri)] = dict(rows=len(rows), known=sum(r["targets"] is not None for r in rows),
                              unknown=sum(r["targets"] is None for r in rows))
    with gzip.open(root / "data/pra_ad/features.pt.gz", "rb") as stream:
        cache = torch.load(io.BytesIO(stream.read()), map_location="cpu", weights_only=True)
    scores = PRAMonitor.from_release().score_features(cache["pra"], cache["counts"])
    if not torch.isfinite(scores).all():
        raise ValueError("Non-finite PRA output")
    return dict(passed=True, checksum_files=len(manifest), checkpoints=4,
                pra_cached_states=len(scores), paired_rounds=counts,
                GPU_calls=0, benchmark_episodes_run=0)
