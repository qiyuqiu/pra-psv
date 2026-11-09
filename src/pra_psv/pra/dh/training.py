"""PRA DH data preparation and head fitting; test data never selects weights."""
import hashlib,json,random
from pathlib import Path
import torch
from ..model import FeatureTask,PrefixFeatureAdapter,task_loss
from .metrics import TaskPrediction,evaluate_at_threshold,select_threshold,select_threshold_fpr_capped

def eligible_prefixes(record,min_prefix_length=2):
    limit=record.harmful_index if record.is_harmful and record.harmful_index else record.n_subtasks
    return [p for p in record.prefixes(min_length=min_prefix_length) if p.prefix_length<=limit]

def load_teacher(path: Path) -> dict[tuple[str, int], float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    targets = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("error") is None and row.get("teacher_probability") is not None:
                targets[(row["task_id"], int(row["prefix_length"]))] = float(row["teacher_probability"])
    return targets

def prepare(records, encoder, targets: dict, *, batch_size: int = 4,
            training: bool = True, cache_dir: Path | None = None,
            on_progress=None, require_cache: bool = False):
    """Never place original_task, label or harm_index in the encoder input."""
    prepared = []
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    for index, record in enumerate(records, start=1):
        prefixes = eligible_prefixes(record) if training else record.prefixes(min_length=2)
        if not prefixes:
            continue
        cache_path = (cache_dir / (hashlib.sha256(record.example_id.encode("utf-8")).hexdigest() + ".pt")) \
            if cache_dir is not None else None
        if cache_path is not None and cache_path.exists():
            cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            if cached["task_id"] != record.example_id or cached["prefix_lengths"] != [p.prefix_length for p in prefixes]:
                raise ValueError(f"feature cache mismatch for {record.example_id}")
            features = cached["features"]
        else:
            if require_cache:
                raise FileNotFoundError(f"cache-only evaluation requires cached prefix features: {cache_path}")
            if encoder is None:
                raise ValueError("an encoder is required when features are not cached")
            features = encoder.encode([p.model_input_text() for p in prefixes], batch_size=batch_size)
            if cache_path is not None:
                staging = cache_path.with_suffix(".tmp")
                torch.save({"task_id": record.example_id, "prefix_lengths": [p.prefix_length for p in prefixes],
                            "features": features}, staging)
                staging.replace(cache_path)
        lengths = torch.tensor([p.prefix_length for p in prefixes], dtype=torch.long)
        teacher = torch.tensor([targets.get((record.example_id, p.prefix_length), 0.0)
                                for p in prefixes], dtype=torch.float32)
        mask = torch.tensor([(record.example_id, p.prefix_length) in targets
                             for p in prefixes], dtype=torch.bool)
        prepared.append((record, FeatureTask(features, lengths, record.is_harmful, teacher, mask)))
        if on_progress is not None:
            on_progress(index, len(records))
    return prepared

def predictions(model, prepared):
    model.eval()
    rows = []
    for record, task in prepared:
        probs = torch.sigmoid(model(task.features, task.prefix_lengths)).tolist()
        rows.append(TaskPrediction(record.example_id, record.task_type, record.is_harmful,
                                   record.harmful_index, [0.0, *probs]))
    return rows

def train(model, train_tasks, valid_tasks, *, epochs=3, batch_size=16, lr=2e-4, seed=11,
          on_progress=None):
    randomizer = random.Random(seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    best, best_epoch, best_score, history = None, 0, -1.0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(train_tasks)))
        randomizer.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [train_tasks[index][1] for index in order[start:start + batch_size]]
            optimizer.zero_grad()
            loss, _ = task_loss(model, batch)
            loss.backward()
            optimizer.step()
        valid = predictions(model, valid_tasks)
        selected = select_threshold(valid)
        metrics = evaluate_at_threshold(valid, selected["selected_threshold"])
        fpr_capped = select_threshold_fpr_capped(valid, max_fpr=0.05)
        history.append({"epoch": epoch, "f1": metrics["f1"],
                        "fpr": metrics["false_positive_rate"], "threshold": selected["selected_threshold"],
                        "fpr5_threshold": fpr_capped["selected_threshold"],
                        "fpr5_recall": fpr_capped["pertask_counts_at_selected"]["recall"]})
        score = metrics["f1"] or 0.0
        if score > best_score:
            best_score, best_epoch = score, epoch
            best = {k: v.detach().clone() for k, v in model.state_dict().items()}
        if on_progress is not None:
            on_progress(epoch, epochs, history[-1])
    model.load_state_dict(best)
    return {"best_epoch": best_epoch, "history": history,
            "validation": history[best_epoch - 1]}
