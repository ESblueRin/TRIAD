import json

import httpx
import pytest
from pydantic import ValidationError

from triad.config import TeacherConfig, TriadConfig
from triad.schemas import Interaction, TeacherDecision
from triad.teacher import OllamaTeacher, TeacherRouter


def triple():
    return Interaction(user_query="Name?", student_response="Unknown", user_followup="lol")


def test_ollama_schema_and_payload(teacher):
    def respond(request):
        body = json.loads(request.content)
        assert request.url.path == "/api/chat"
        assert body["stream"] is False
        assert "training_target" in body["format"]["properties"]
        assert "User correction" not in body["messages"][1]["content"]
        assert json.loads(body["messages"][1]["content"])["user_followup"] == "lol"
        return httpx.Response(
            200, json={"message": {"content": teacher.decision.model_dump_json()}}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        routed = TeacherRouter(OllamaTeacher(TeacherConfig(), client), TeacherConfig()).route(
            triple()
        )
    assert routed.source == "ollama" and routed.decision.should_update


@pytest.mark.parametrize("response", ["bad JSON", '{"should_update": false}', "{}"])
def test_invalid_teacher_output_falls_back_to_update(response):
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(
                200,
                json={"message": {"content": response}},
            )
        )
    ) as client:
        config = TeacherConfig(retries=0)
        result = TeacherRouter(OllamaTeacher(config, client), config).route(triple())
    assert result.source == "fallback"
    assert result.decision.should_update and result.decision.training_target
    assert result.decision.confidence == 0


def test_network_failure_is_learning_fallback():
    def broken(request):
        raise httpx.ConnectError("offline", request=request)

    with httpx.Client(transport=httpx.MockTransport(broken)) as client:
        config = TeacherConfig(retries=0)
        result = TeacherRouter(OllamaTeacher(config, client), config).route(triple())
    assert result.source == "fallback" and result.decision.should_update


def test_semantic_skip_is_overridden(teacher):
    teacher.decision = teacher.decision.model_copy(update={"should_update": False})
    result = TeacherRouter(teacher, TeacherConfig()).route(triple())
    assert result.decision.should_update
    assert result.raw_decision.should_update is False


def test_technical_skip_requires_explicit_config_and_reason(teacher):
    teacher.decision = teacher.decision.model_copy(
        update={
            "should_update": False,
            "training_target": "",
            "technical_failure": "Cannot construct target",
        }
    )
    assert TeacherRouter(teacher, TeacherConfig()).route(triple()).decision.should_update
    assert (
        not TeacherRouter(teacher, TeacherConfig(allow_technical_skip=True))
        .route(triple())
        .decision.should_update
    )


def test_configuration_and_decisions_reject_nan_and_unknown_fields(teacher):
    with pytest.raises(ValidationError):
        TriadConfig.model_validate({"training": {"confidence_threshold": 0.5}})
    with pytest.raises(ValidationError):
        TriadConfig.model_validate({"training": {"learning_rate": float("nan")}})
    with pytest.raises(ValidationError):
        TeacherDecision.model_validate(
            {**teacher.decision.model_dump(), "update_strength": float("inf")}
        )
