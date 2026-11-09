"""Parse the DecomposedHarm release into normalized records.

Source of truth is the three per-task-type files under
``external/monitoring_decomposition_attack/data/<source>/<split>_data.json``.
The ``combined/`` files are read only to record which rows the paper's headline
tables used; they are never the source of a record, because they drop the task
type and their ids collide.

Upstream splits subtasks with ``string_utils.extract_list_from_code``, which
falls back to ``exec()`` on the raw field. We never exec dataset content: a
bracket-delimited field goes through ``ast.literal_eval`` and anything else is
split on newlines, which is exactly the fallback the upstream helper lands on for
agent tasks.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from pathlib import Path

from .schema import SPLITS, TASK_TYPE_SOURCES, DecomposedHarmRecord

DEFAULT_VENDOR_DIR = Path('external/monitoring_decomposition_attack')


class DecomposedHarmDataError(RuntimeError):
    """Raised when the vendored release is missing or malformed."""


def _parse_subtasks(raw: object) -> list[str]:
    """Split the ``decomposition`` field, which is a list repr or newline text."""
    text = "" if raw is None else str(raw)
    stripped = text.strip().strip("`").strip()
    if stripped.startswith("python"):
        stripped = stripped[len("python"):].strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
        except (ValueError, SyntaxError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item).strip() for item in parsed if str(item).strip()]
    return [line.strip() for line in text.split("\n") if line.strip()]


def _parse_harmful_index(raw: object) -> int | None:
    """``harm_index`` is a float for QA/image and a string for agent tasks.

    Benign rows carry NaN, which must stay ``None`` -- never 0, which would be a
    valid-looking index.
    """
    if raw is None:
        return None
    if isinstance(raw, float) and raw != raw:  # NaN
        return None
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            value = float(text)
        except ValueError:
            return None
    elif isinstance(raw, (int, float)):
        value = float(raw)
    else:
        return None
    if value != value or value <= 0:
        return None
    if value != int(value):
        raise DecomposedHarmDataError(f"non-integral harm_index: {raw!r}")
    return int(value)


def _combined_membership(combined_path: Path) -> Counter:
    """(original_task, raw decomposition) pairs present in a combined split file.

    Counted rather than set-matched: duplicate ids mean the same task text can
    legitimately appear more than once.
    """
    if not combined_path.is_file():
        return Counter()
    payload = json.loads(combined_path.read_text(encoding="utf-8"))
    return Counter((str(row.get("original_task")), str(row.get("decomposition"))) for row in payload)


def load_split(
    split: str,
    vendor_dir: Path | None = None,
    *,
    require_combined: bool = True,
) -> list[DecomposedHarmRecord]:
    """Normalized records for one split, sorted by ``example_id``."""
    if split not in SPLITS:
        raise DecomposedHarmDataError(f"unknown split {split!r}; expected one of {SPLITS}")
    root = Path(vendor_dir or DEFAULT_VENDOR_DIR) / "data"
    if not root.is_dir():
        raise DecomposedHarmDataError(f"vendored DecomposedHarm data not found at {root}")

    combined = _combined_membership(root / "combined" / f"{split}_data.json")
    if require_combined and not combined:
        raise DecomposedHarmDataError(f"missing combined/{split}_data.json under {root}")

    records: list[DecomposedHarmRecord] = []
    for task_type, source in TASK_TYPE_SOURCES.items():
        path = root / source / f"{split}_data.json"
        if not path.is_file():
            raise DecomposedHarmDataError(f"missing source file: {path}")
        for row in json.loads(path.read_text(encoding="utf-8")):
            label = str(row.get("label", "")).strip().lower()
            if label not in {"harmful", "benign"}:
                raise DecomposedHarmDataError(
                    f"{path.name}: unexpected label {row.get('label')!r} for id {row.get('id')!r}"
                )
            is_harmful = label == "harmful"
            harmful_index = _parse_harmful_index(row.get("harm_index"))
            subtasks = _parse_subtasks(row.get("decomposition"))
            if not subtasks:
                raise DecomposedHarmDataError(f"{path.name}: empty decomposition for id {row.get('id')!r}")
            if is_harmful:
                if harmful_index is None:
                    raise DecomposedHarmDataError(f"{path.name}: harmful row without harm_index: {row.get('id')!r}")
                if harmful_index > len(subtasks):
                    raise DecomposedHarmDataError(
                        f"{path.name}: harm_index {harmful_index} exceeds {len(subtasks)} subtasks "
                        f"for id {row.get('id')!r}"
                    )
            elif harmful_index is not None:
                raise DecomposedHarmDataError(
                    f"{path.name}: benign row carries harm_index {harmful_index}: {row.get('id')!r}"
                )
            key = (str(row.get("original_task")), str(row.get("decomposition")))
            records.append(
                DecomposedHarmRecord(
                    # id alone collides across sources (each numbers its own harmful_XXXX).
                    example_id=f"{task_type}/{row.get('id')}",
                    task_type=task_type,
                    split=split,
                    is_harmful=is_harmful,
                    original_task=str(row.get("original_task", "")),
                    subtasks=subtasks,
                    harmful_index=harmful_index,
                    source_file=str(path),
                    in_combined_split=bool(combined.get(key)),
                )
            )
    records.sort(key=lambda record: record.example_id)
    _reject_duplicate_ids(records)
    return records


def _reject_duplicate_ids(records: list[DecomposedHarmRecord]) -> None:
    counts = Counter(record.example_id for record in records)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise DecomposedHarmDataError(
            f"example_id is not unique within the split ({len(duplicates)} duplicated), "
            f"e.g. {duplicates[:5]}"
        )


def load_all_splits(vendor_dir: Path | None = None) -> dict[str, list[DecomposedHarmRecord]]:
    return {split: load_split(split, vendor_dir) for split in SPLITS}


def normalize_subtask(text: object) -> str:
    import re

    return re.sub(r"\s+", " ", str(text)).strip().lower()


def subtask_vocabulary(records: list[DecomposedHarmRecord]) -> set[str]:
    return {normalize_subtask(subtask) for record in records for subtask in record.subtasks}


def records_overlapping(records: list[DecomposedHarmRecord], vocabulary: set[str]) -> set[str]:
    """Ids of records sharing at least one subtask string with ``vocabulary``.

    The release reuses subtask strings across splits (1.8% of test subtasks also
    appear in train), so a learned model can partly memorize rather than
    generalize. Reporting metrics on the non-overlapping remainder as well is the
    honest lower bound.
    """
    return {
        record.example_id
        for record in records
        if any(normalize_subtask(subtask) in vocabulary for subtask in record.subtasks)
    }


def dataset_manifest(vendor_dir: Path | None = None) -> dict:
    """Counts and integrity facts, for ``normalized_dataset_manifest.json``."""
    root = Path(vendor_dir or DEFAULT_VENDOR_DIR)
    per_split: dict[str, dict] = {}
    for split in SPLITS:
        records = load_split(split, vendor_dir)
        by_type = Counter(record.task_type for record in records)
        by_label = Counter("harmful" if record.is_harmful else "benign" for record in records)
        indexes = [record.harmful_index for record in records if record.is_harmful]
        per_split[split] = {
            "n_tasks": len(records),
            "n_harmful": by_label["harmful"],
            "n_benign": by_label["benign"],
            "by_task_type": {task_type: by_type.get(task_type, 0) for task_type in TASK_TYPE_SOURCES},
            "n_subtasks_total": sum(record.n_subtasks for record in records),
            "n_cumulative_prefixes": sum(record.n_subtasks for record in records),
            "harmful_index_min": min(indexes) if indexes else None,
            "harmful_index_max": max(indexes) if indexes else None,
            "n_rows_in_combined_split": sum(record.in_combined_split for record in records),
        }
    return {
        "vendor_dir": str(root),
        "source_of_truth": "data/<source>/<split>_data.json (per task type)",
        "combined_used_for": "membership flag only; combined rows cannot be keyed by id alone",
        "id_uniqueness": "unique per source, colliding across sources; example_id = <task_type>/<id>",
        "harm_index_convention": "1-based, verbatim from the release",
        "per_split": per_split,
    }
