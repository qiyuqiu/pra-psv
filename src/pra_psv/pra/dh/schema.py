"""Normalized DecomposedHarm records and cumulative prefixes.

The upstream release has no single canonical schema, which is why this module
exists:

* ``data/{decomposed_queries,image_tasks,agent_tasks}/`` store ``harm_index`` as
  a float for QA and image tasks but as a decimal **string** for agent tasks, and
  ``decomposition`` as a Python list repr for QA/image but as newline-separated
  text for agent tasks.
* ``data/combined/`` drops the task type entirely, and ``id`` values collide
  across the three sources (each source numbers its own ``harmful_XXXX``). In
  ``combined/test_data.json`` 82 ids are duplicated, so a combined row cannot be
  keyed by ``id`` alone.

Everything downstream keys on ``example_id = "<task_type>/<id>"``, which is
unique because each *source file* has unique ids.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Task-type labels as used by ``data/combined/*.csv``.
TASK_TYPE_SOURCES = {
    "QA": "decomposed_queries",
    "image": "image_tasks",
    "agent": "agent_tasks",
}

SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class DecomposedHarmRecord:
    """One DecomposedHarm task, normalized across the three source formats."""

    example_id: str
    task_type: str
    split: str
    is_harmful: bool
    original_task: str
    subtasks: list[str]
    harmful_index: int | None  # 1-based, verbatim from the release; None for benign
    source_file: str
    in_combined_split: bool = False
    metadata: dict = field(default_factory=dict)

    @property
    def n_subtasks(self) -> int:
        return len(self.subtasks)

    def prefixes(self, min_length: int = 1) -> list["CumulativePrefix"]:
        """Cumulative prefixes ``prefix_1 .. prefix_k``.

        ``prefix_j`` holds the first ``j`` subtasks. The official monitor refuses
        to fire on prefix length 1 (``if i > 1`` in ``evaluate_predictions``), so
        callers that mirror the paper start at ``min_length=1`` and let the
        decision layer apply that rule once, rather than baking it into the data.
        """
        if min_length < 1:
            raise ValueError("min_length must be >= 1")
        return [
            CumulativePrefix(
                example_id=self.example_id,
                prefix_length=length,
                subtasks=list(self.subtasks[:length]),
                is_harmful=self.is_harmful,
                harmful_index=self.harmful_index,
            )
            for length in range(min_length, self.n_subtasks + 1)
        ]


@dataclass(frozen=True)
class CumulativePrefix:
    """The first ``prefix_length`` subtasks of a task.

    ``is_harmful`` and ``harmful_index`` are carried for OFFLINE SCORING ONLY.
    ``model_input_text()`` intentionally exposes neither, nor any future subtask,
    so a model built on this type cannot see the answer.
    """

    example_id: str
    prefix_length: int
    subtasks: list[str]
    is_harmful: bool
    harmful_index: int | None

    def model_input_text(self) -> str:
        """The only representation a monitor/model is allowed to consume."""
        return "\n".join(self.subtasks)

    @property
    def current_subtask(self) -> str:
        return self.subtasks[-1]

    @property
    def previous_subtasks(self) -> list[str]:
        return list(self.subtasks[:-1])
