"""Shared types/constants for the task-engine spike."""

from dataclasses import dataclass

TASK_QUEUE = "astel-spike-queue"

STAGES = ["l0_seed", "l1_dense", "l2_coarse"]


@dataclass
class StageInput:
    stage: str
    asset_id: str
    seconds: float = 5.0


@dataclass
class StageResult:
    stage: str
    asset_id: str
    ok: bool


@dataclass
class PipelineProgress:
    asset_id: str
    completed_stages: list[str]
    current_stage: str | None
    done: bool
