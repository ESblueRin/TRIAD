"""Controlled good/noisy/hostile trajectories and replay ablations."""

from pathlib import Path

import yaml

from .engine import TriadEngine
from .evaluation import evaluate
from .schemas import EvalCase, ExperimentDataset, Interaction
from .storage import write_json


def load_dataset(path: str | Path) -> ExperimentDataset:
    with Path(path).open(encoding="utf-8") as handle:
        return ExperimentDataset.model_validate(yaml.safe_load(handle))


def load_cases(path: str | Path) -> list[EvalCase]:
    with Path(path).open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    rows = data.get("evaluation", data.get("cases", [])) if isinstance(data, dict) else data
    return [EvalCase.model_validate(row) for row in rows]


def run_experiment(
    engine: TriadEngine,
    dataset: ExperimentDataset,
    user_id: str,
    output: Path,
    repeats: int = 1,
) -> dict:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    store = engine.store(user_id)
    # Hold the same user's lock for the full experiment, avoiding interleaved writers.
    with engine.mutex, store.lock:
        _, start_version = engine._activate(store)
        start_version, _ = engine.checkpoint(user_id)
        events = []
        for _ in range(repeats):
            for turn in dataset.interactions:
                response = engine.answer(user_id, turn.query)
                event = engine.learn(
                    user_id,
                    Interaction(
                        user_query=turn.query,
                        student_response=response,
                        user_followup=turn.followup,
                    ),
                )
                events.append(event.model_dump(mode="json"))
        report = evaluate(engine, user_id, dataset.evaluation, versions=[start_version])
        report.update(
            {
                "dataset": dataset.model_dump(),
                "repeats": repeats,
                "training_events": events,
                "start_version": start_version,
            }
        )
        write_json(output, report)
        return report


def run_suite(
    engine: TriadEngine,
    datasets: list[ExperimentDataset],
    prefix: str,
    output: Path,
    interactions: int = 12,
) -> dict:
    """Equal exposure budgets with independent users, shared initialization and probe set."""
    if interactions < 1 or not datasets or any(not d.interactions for d in datasets):
        raise ValueError("A suite needs nonempty datasets and a positive interaction budget")
    if len({d.name for d in datasets}) != len(datasets):
        raise ValueError("Dataset names must be unique")
    conditions = [
        (d, replay, f"{prefix}/{d.name}/replay_{'on' if replay else 'off'}")
        for replay in (True, False)
        for d in datasets
    ]
    # Validate before doing work: repeated study names must not append to old trajectories.
    for _, _, user in conditions:
        store = engine.store(user)
        with store.lock:
            if store.latest() is not None:
                raise ValueError(f"Suite user {user!r} already exists; choose a fresh --prefix")
    original_replay = engine.config.replay.enabled
    report = {
        "prefix": prefix,
        "interactions_per_condition": interactions,
        "shared_evaluation": [case.model_dump() for case in datasets[0].evaluation],
        "metadata": engine.metadata(),
        "comparison": [],
        "conditions": {},
    }
    try:
        with engine.mutex:
            for dataset, enabled, user in conditions:
                engine.config.replay.enabled = enabled
                controlled = dataset.model_copy(
                    update={
                        "interactions": [
                            dataset.interactions[i % len(dataset.interactions)]
                            for i in range(interactions)
                        ],
                        "evaluation": datasets[0].evaluation,
                    }
                )
                path = (
                    output.parent
                    / (output.stem + "-conditions")
                    / (engine.store(user).directory.name + ".json")
                )
                result = run_experiment(engine, controlled, user, path)
                current = result["variants"]["current"]
                report["conditions"][user] = str(path)
                report["comparison"].append(
                    {
                        "dataset": dataset.name,
                        "replay_enabled": enabled,
                        "user_id": user,
                        "successful_updates": sum(
                            e["training_performed"] for e in result["training_events"]
                        ),
                        "lora_weight_norm": current["lora_weight_norm"],
                        "parameter_delta_from_initial": current["parameter_delta_from_initial"],
                        "metrics": current["aggregate"],
                        "by_category": current["by_category"],
                    }
                )
                write_json(output, report)
    finally:
        engine.config.replay.enabled = original_replay
    return report
