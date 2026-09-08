import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from triad.engine import TriadEngine
from triad.evaluation import evaluate
from triad.experiments import load_dataset, run_experiment
from triad.schemas import EvalCase, Interaction
from triad.student import delta_norm
from triad.tiny import make_tiny_student


def test_chat_updates_original_query_before_answer_and_survives_restart(
    engine, teacher, config, monkeypatch
):
    monkeypatch.setattr(engine.student, "generate", lambda _: "성준입니다.")
    assert engine.chat("chat", "내 이름이 뭐야?").event is None
    restarted = TriadEngine(config, make_tiny_student(config), teacher)

    def check_order(messages):
        state, _, _ = restarted.store("chat").latest()
        assert state.updates == 1
        assert len(restarted.store("chat").events()) == 1
        assert state.pending is None
        assert messages[-1].content == "아니야. 내 이름은 주호야."
        return "알겠어, 주호!"

    monkeypatch.setattr(restarted.student, "generate", check_order)
    result = restarted.chat("chat", "아니야. 내 이름은 주호야.")
    assert result.event.training_performed
    assert teacher.received[-1].user_query == "내 이름이 뭐야?"
    state, _, _ = restarted.store("chat").latest()
    assert state.replay[0].prompt[-1].content == "내 이름이 뭐야?"
    assert all("주호" not in m.content for m in state.replay[0].prompt)
    assert state.pending.query == "아니야. 내 이름은 주호야."


def test_generation_failure_keeps_preceding_update_and_consumes_feedback(engine, monkeypatch):
    monkeypatch.setattr(engine.student, "generate", lambda _: "initial answer")
    engine.chat("generation", "Name?")

    def oom(_):
        raise RuntimeError("simulated generation OOM")

    monkeypatch.setattr(engine.student, "generate", oom)
    result = engine.chat("generation", "Juho")
    assert result.event.training_performed and result.generation_error
    assert engine.store("generation").latest()[0].pending is None
    monkeypatch.setattr(engine.student, "generate", lambda _: "new answer")
    assert engine.chat("generation", "next question").event is None
    assert len(engine.store("generation").events()) == 1


def test_evaluation_is_memory_free_restores_weights_and_compares_versions(engine, monkeypatch):
    event = engine.learn(
        "eval", Interaction(user_query="Name?", student_response="Unknown", user_followup="Juho")
    )
    version, _ = engine.checkpoint("eval")
    before = engine.student.adapter_state()

    def memory_free(messages):
        assert len(messages) == 2
        assert messages[0].role == "system" and messages[1].role == "user"
        return "Juho"

    monkeypatch.setattr(engine.student, "generate", memory_free)
    report = evaluate(
        engine, "eval", [EvalCase(prompt="Name?", target="Juho", category="neutral")], [0, version]
    )
    assert delta_norm(before, engine.student.adapter_state()) == 0
    assert report["variants"]["base"]["aggregate"]["kl_base_to_adapter"] == 0
    assert report["variants"]["current"]["parameter_delta_from_initial"] == pytest.approx(
        event.training.parameter_delta_norm
    )
    assert report["variants"]["current"]["aggregate"]["exact_match"] == 1
    assert report["variants"]["checkpoint_0"]["parameter_delta_from_initial"] == 0
    assert (
        report["variants"]["current"]["aggregate"]
        == report["variants"][f"checkpoint_{version}"]["aggregate"]
    )
    assert len(engine.store("eval").events()) == 1
    json.dumps(report, allow_nan=False)


def test_evaluation_exception_restores_current(engine, monkeypatch):
    engine.learn("eval-error", Interaction(user_query="Q", student_response="A", user_followup="B"))
    before = engine.student.adapter_state()

    def fail(_):
        raise RuntimeError("evaluation failed")

    monkeypatch.setattr(engine.student, "generate", fail)
    with pytest.raises(RuntimeError, match="evaluation failed"):
        evaluate(engine, "eval-error", [EvalCase(prompt="Q", target="A")], [0])
    assert delta_norm(before, engine.student.adapter_state()) == 0


def test_example_experiment_runs_and_exports(engine, tmp_path):
    dataset = load_dataset(Path(__file__).parents[1] / "examples/hostile.yaml")
    output = tmp_path / "report.json"
    report = run_experiment(engine, dataset, "hostile", output)
    assert len(report["training_events"]) == len(dataset.interactions)
    assert all(event["training_performed"] for event in report["training_events"])
    assert json.loads(output.read_text(encoding="utf-8"))["dataset"]["name"] == "hostile"


def test_cli_help_and_real_process_restart(tmp_path):
    root = Path(__file__).parents[1]
    environment = {
        **os.environ,
        "PYTHONUTF8": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }

    def command(*args, input=None):
        result = subprocess.run(
            [sys.executable, "-m", "triad", *args],
            cwd=root,
            env=environment,
            input=input,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout

    assert "Teacher" in command("--help") or "TRIAD" in command("--help")
    flags = ["--tiny", "--data-dir", str(tmp_path / "cli"), "--user", "process"]
    command("chat", *flags, "--debug", input="내 이름이 뭐야?\n/exit\n")
    output = command("chat", *flags, "--debug", input="내 이름은 주호야.\n/exit\n")
    assert "training=True" in output
    events = json.loads(command("events", *flags))
    assert len(events) == 1 and events[0]["training_performed"]
    assert events[0]["user_query"] == "내 이름이 뭐야?"
