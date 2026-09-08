"""Bounded FIFO replay with deterministic sampling; stored per user and checkpoint."""

import random

from .config import ReplayConfig
from .schemas import TrainingExample


def sample_replay(
    entries: list[TrainingExample],
    config: ReplayConfig,
    seed: int,
) -> list[TrainingExample]:
    if not config.enabled:
        return []
    return random.Random(seed).sample(entries, min(config.samples_per_interaction, len(entries)))


def append_replay(
    entries: list[TrainingExample],
    example: TrainingExample,
    config: ReplayConfig,
) -> list[TrainingExample]:
    # Keep the data even when sampling is disabled, so ON/OFF can be changed later.
    return (entries + [example])[-config.capacity :]
