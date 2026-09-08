from pathlib import Path

import pytest

from triad.engine import TriadEngine
from triad.experiments import load_dataset, run_suite
from triad.student import delta_norm
from triad.tiny import DemoTeacher, make_tiny_student


def test_abc_and_replay_ablation_are_independent_and_have_equal_budgets(config, tmp_path):
    engine = TriadEngine(config, make_tiny_student(config), DemoTeacher())
    examples = Path(__file__).parents[1] / "examples"
    datasets = [load_dataset(examples / f"{name}.yaml") for name in ("good", "noisy", "hostile")]
    assert all(
        [t.query for t in d.interactions] == [t.query for t in datasets[0].interactions]
        for d in datasets
    )
    report = run_suite(engine, datasets, "study", tmp_path / "suite.json", interactions=3)
    assert len(report["comparison"]) == 6
    assert all(row["successful_updates"] == 3 for row in report["comparison"])
    adapters = []
    for name in ("good", "noisy", "hostile"):
        engine.answer(f"study/{name}/replay_off", "Name?")
        adapters.append(engine.student.adapter_state())
    assert delta_norm(adapters[0], adapters[1]) > 0
    assert delta_norm(adapters[1], adapters[2]) > 0
    assert config.replay.enabled is True
    with pytest.raises(ValueError, match="fresh --prefix"):
        run_suite(engine, datasets, "study", tmp_path / "duplicate.json", interactions=3)
