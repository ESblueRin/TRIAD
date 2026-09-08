"""Public records used by routing, training, persistence and experiments."""

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, revalidate_instances="always")


class Message(Record):
    role: Literal["system", "user", "assistant"]
    content: str


class Interaction(Record):
    user_query: str
    student_response: str
    user_followup: str
    context: list[Message] = Field(default_factory=list)


class TeacherDecision(Record):
    feedback_type: Literal[
        "corrective", "positive", "negative", "ambiguous", "noise", "hostile", "other"
    ]
    confidence: float = Field(ge=0, le=1)
    interpretation: str = Field(min_length=1, max_length=600)
    training_target: str = Field(max_length=12000)
    update_strength: float = Field(ge=0, le=1)
    should_update: bool = True
    technical_failure: str | None = Field(default=None, max_length=600)

    @model_validator(mode="after")
    def meaningful_fields(self):
        if not self.interpretation.strip():
            raise ValueError("interpretation must contain text")
        if self.should_update and not self.training_target.strip():
            raise ValueError("an update requires a nonempty training_target")
        return self


class RoutedDecision(Record):
    decision: TeacherDecision
    source: Literal["ollama", "fallback", "injected"]
    raw_decision: TeacherDecision | None = None
    error: str | None = None


class TrainingExample(Record):
    event_id: str
    prompt: list[Message]
    target: str = Field(min_length=1)
    strength: float = Field(gt=0, le=1)


class TrainingResult(Record):
    performed: bool
    loss: float | None = None
    losses: list[float] = Field(default_factory=list)
    gradient_norm: float | None = None
    parameter_delta_norm: float = 0.0
    effective_learning_rate: float = 0.0
    replay_count: int = 0
    target_tokens: int = 0
    seed: int | None = None
    error: str | None = None


class PendingTurn(Record):
    query: str
    response: str
    context: list[Message] = Field(default_factory=list)


class UserState(Record):
    user_id: str
    signature: str
    created_at: str = Field(default_factory=utc_now)
    updates: int = 0
    interactions: int = 0
    history: list[Message] = Field(default_factory=list)
    pending: PendingTurn | None = None
    replay: list[TrainingExample] = Field(default_factory=list)
    last_event_id: str | None = None


class TrainingEvent(Record):
    event_id: str
    timestamp: str = Field(default_factory=utc_now)
    user_id: str
    user_query: str
    student_response: str
    user_followup: str
    context: list[Message]
    teacher_feedback_type: str
    teacher_interpretation: str
    teacher_target: str
    teacher_confidence: float
    teacher_decision: RoutedDecision
    update_strength: float
    effective_update_strength: float
    training_performed: bool
    loss: float | None
    gradient_norm: float | None
    training: TrainingResult
    adapter_checkpoint: str | None = None
    adapter_version: int | None = None
    replay_event_ids: list[str] = Field(default_factory=list)
    config: dict


class EvalCase(Record):
    prompt: str = Field(min_length=1)
    target: str = Field(min_length=1)
    category: str = "neutral"


class ExperimentTurn(Record):
    query: str = Field(min_length=1)
    followup: str


class ExperimentDataset(Record):
    name: str
    interactions: list[ExperimentTurn]
    evaluation: list[EvalCase]
