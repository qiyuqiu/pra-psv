"""Repository assets and checked CPU checkpoint loading."""
import gzip
import hashlib
import io
import json
import os
from pathlib import Path


def release_root():
    root = Path(os.environ.get("PRA_PSV_ROOT", Path(__file__).resolve().parents[2]))
    if not (root / "artifacts/checkpoints.json").is_file():
        raise FileNotFoundError("Set PRA_PSV_ROOT to the extracted release repository")
    return root


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checkpoint(name):
    import torch
    root = release_root()
    entry = json.loads((root / "artifacts/checkpoints.json").read_text())[name]
    path = root / entry["path"]
    if sha256(path) != entry["sha256"]:
        raise ValueError(f"Checkpoint checksum mismatch: {name}")
    with gzip.open(path, "rb") as stream:
        return torch.load(io.BytesIO(stream.read()), map_location="cpu", weights_only=True)


def pairs(round_index):
    if round_index not in (0, 1):
        raise ValueError("round must be 0 or 1")
    files = sorted((release_root() / f"data/psv/round{round_index}").glob("part-*.jsonl"))
    if not files:
        raise FileNotFoundError("Missing cached pair shards")
    return [json.loads(line) for path in files for line in path.read_text().splitlines() if line.strip()]

