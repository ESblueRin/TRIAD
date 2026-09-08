"""Opt-in: pytest -m integration. Never contacted during the regular test suite."""

import os

import pytest

from triad.config import TriadConfig
from triad.engine import TriadEngine
from triad.schemas import Interaction
from triad.student import Student, tensor_digest
from triad.teacher import OllamaTeacher
from triad.tiny import make_tiny_student, tiny_config

pytestmark = pytest.mark.integration


def test_downloaded_transformer_real_peft(tmp_path, teacher):
    model_name = os.environ.get("TRIAD_INTEGRATION_MODEL")
    if not model_name:
        pytest.skip("Set TRIAD_INTEGRATION_MODEL to opt into a Hugging Face download")
    config = TriadConfig.model_validate(
        {
            "student": {
                "model_name": model_name,
                "device": "cpu",
                "max_length": 128,
                "max_new_tokens": 8,
            },
            "storage": {"root": str(tmp_path)},
        }
    )
    student = Student.from_config(config)
    frozen = {n: p for n, p in student.model.named_parameters() if "lora_" not in n}
    digest = tensor_digest(frozen)
    engine = TriadEngine(config, student, teacher)
    event = engine.learn(
        "integration",
        Interaction(user_query="Name?", student_response="Unknown", user_followup="Juho"),
    )
    assert event.training_performed, event.training.error
    assert tensor_digest(frozen) == digest


def test_live_ollama_routes_uninformative_abuse(tmp_path):
    model_name = os.environ.get("TRIAD_OLLAMA_MODEL")
    if not model_name:
        pytest.skip("Set TRIAD_OLLAMA_MODEL to opt into a running local Ollama model")
    config = tiny_config(tmp_path / "live")
    config.teacher.model_name = model_name
    config.teacher.timeout_seconds = 120
    engine = TriadEngine(config, make_tiny_student(config), OllamaTeacher(config.teacher))
    frozen = {n: p for n, p in engine.student.model.named_parameters() if "lora_" not in n}
    digest = tensor_digest(frozen)
    assert engine.chat("live", "내 이름이 뭐야?").event is None
    result = engine.chat("live", "ㅅㅂ 진짜 답답하네.")
    assert result.event.teacher_decision.source == "ollama", result.event.teacher_decision.error
    assert result.event.training_performed, result.event.training.error
    assert engine.store("live").latest()[0].updates == 1
    assert tensor_digest(frozen) == digest
