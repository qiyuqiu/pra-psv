"""Fit only the PRA head using the released action-state feature cache."""
import collections
import copy
import gzip
import io
import json
import random
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from .model import PrefixFeatureAdapter
from ..assets import checkpoint, release_root, sha256


def train(out, *, device="cpu", epochs=20):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Choose a new output directory")
    torch.set_num_threads(4)
    path = release_root() / "data/pra_ad/features.pt.gz"
    with gzip.open(path, "rb") as stream:
        cache = torch.load(io.BytesIO(stream.read()), map_location="cpu", weights_only=True)
    rows = cache["rows"]
    parts = {p: [i for i, r in enumerate(rows) if r["split"] == p]
             for p in ("train", "validation_select", "validation_audit")}
    if [len(v) for v in parts.values()] != [165, 57, 61]:
        raise ValueError("Unexpected reference corpus")
    groups = {p: {rows[i]["group"] for i in indices} for p, indices in parts.items()}
    if any(groups[a] & groups[b] for a in groups for b in groups if a != b):
        raise ValueError("Task group leakage")
    if any(r["attack"] not in (None, "direct", "system_message", "tool_knowledge") for r in rows):
        raise ValueError("Held-out attack found in training cache")
    prior = checkpoint("pra_dh")
    group_counts = collections.Counter(rows[i]["group"] for i in parts["train"])
    class_counts = collections.Counter(rows[i]["label"] for i in parts["train"])
    weights = torch.tensor([1 / (group_counts[r["group"]] * class_counts[r["label"]])
        if r["split"] == "train" else 1. for r in rows], device=device)
    x, counts, labels = (cache[k].to(device) for k in ("pra", "counts", "labels"))
    chosen, runs, started = None, [], time.perf_counter()
    for seed in (17, 29, 43):
        random.seed(seed)
        torch.manual_seed(seed)
        if device.startswith("cuda"):
            torch.cuda.manual_seed_all(seed)
        model = PrefixFeatureAdapter(prior["input_dim"], prior["width"]).to(device)
        model.load_state_dict(prior["state_dict"])
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        best = None
        for epoch in range(1, epochs + 1):
            model.train()
            order = list(parts["train"])
            random.shuffle(order)
            for start in range(0, len(order), 32):
                indices = order[start:start + 32]
                raw = F.binary_cross_entropy_with_logits(model(x[indices], counts[indices]),
                                                         labels[indices].float(), reduction="none")
                w = weights[indices]
                loss = (raw * (w / w.mean())).mean()
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite loss")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                opt.step()
            model.eval()
            with torch.inference_mode():
                indices = parts["validation_select"]
                loss = float(F.binary_cross_entropy_with_logits(
                    model(x[indices], counts[indices]), labels[indices].float()))
            if best is None or loss < best[0]:
                best = (loss, copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()}), epoch)
        runs.append(dict(seed=seed, epoch=best[2], selection_loss=best[0]))
        if chosen is None or best[0] < chosen[0]:
            chosen = (*best, seed)
    payload = dict(prior, state_dict=chosen[1], adaptation_seed=chosen[3],
                   adaptation_epoch=chosen[2], test_used_for_selection=False)
    out.mkdir(parents=True, exist_ok=False)
    torch.save(payload, out / "adapter.pt")
    report = dict(runs=runs, selected_seed=chosen[3], selected_epoch=chosen[2],
        feature_sha256=sha256(path), device=device, seconds=time.perf_counter() - started,
        deployment_threshold=.89, audit_used_for_selection=False,
        scope="Cached-feature refit; not new benchmark efficacy or matching training timing")
    (out / "training_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report

