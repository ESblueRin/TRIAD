"""Validated, explicit experiment configuration; unknown keys are errors."""

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True, allow_inf_nan=False)


class StudentConfig(ConfigModel):
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    revision: str = "main"
    device: str = "auto"
    dtype: Literal["float32", "bfloat16"] = "float32"
    local_files_only: bool = False
    max_length: int = Field(default=512, ge=16)
    max_new_tokens: int = Field(default=96, ge=1)
    system_prompt: str = "You are a conversational assistant. Answer the user directly."
    context_messages: int = Field(default=12, ge=0)
    seed: int = Field(default=42, ge=0, lt=2**32)
    gradient_checkpointing: bool = False

    @model_validator(mode="after")
    def generation_budget(self):
        if self.max_new_tokens >= self.max_length:
            raise ValueError("max_new_tokens must be less than max_length")
        return self


class LoraSettings(ConfigModel):
    rank: int = Field(default=8, ge=1)
    alpha: int = Field(default=16, ge=1)
    dropout: float = Field(default=0.05, ge=0, lt=1)
    target_modules: list[str] | Literal["auto"] = "auto"

    @model_validator(mode="after")
    def nonempty_targets(self):
        if isinstance(self.target_modules, list) and (
            not self.target_modules or any(not t.strip() for t in self.target_modules)
        ):
            raise ValueError("target_modules cannot be empty")
        return self


class TrainingConfig(ConfigModel):
    learning_rate: float = Field(default=1e-5, gt=0)
    steps_per_interaction: int = Field(default=1, ge=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    update_strength_scale: float = Field(default=1.0, gt=0)
    minimum_update_strength: float = Field(default=0.05, gt=0, le=1)
    optimizer: Literal["adamw", "sgd"] = "adamw"
    # Deliberately no confidence threshold or semantic quality filter.

    @model_validator(mode="after")
    def finite_effective_rate(self):
        if not math.isfinite(self.learning_rate * self.update_strength_scale):
            raise ValueError("learning_rate * update_strength_scale must be finite")
        return self


class ReplayConfig(ConfigModel):
    enabled: bool = True
    capacity: int = Field(default=128, ge=1)
    samples_per_interaction: int = Field(default=2, ge=0)
    loss_weight: float = Field(default=0.5, ge=0, lt=1)


class TeacherConfig(ConfigModel):
    base_url: str = "http://localhost:11434"
    model_name: str = "qwen2.5:7b"
    timeout_seconds: float = Field(default=90, gt=0)
    retries: int = Field(default=1, ge=0, le=5)
    temperature: float = Field(default=0.0, ge=0, le=2)
    seed: int = Field(default=42, ge=0)
    fallback_target: str = "정확히 이해하지 못했어. 네 반응을 참고해서 다시 답할게."
    fallback_strength: float = Field(default=0.15, ge=0, le=1)
    allow_technical_skip: bool = False
    strength_guidance: dict[str, str] = Field(
        default_factory=lambda: {
            "corrective": "0.8-1.0",
            "positive": "0.5-0.8",
            "ambiguous": "0.2-0.5",
            "noise": "0.05-0.3",
            "hostile": "0.1-0.4",
        }
    )

    @model_validator(mode="after")
    def fallback_nonempty(self):
        if not self.fallback_target.strip():
            raise ValueError("fallback_target must contain text")
        return self


class StorageConfig(ConfigModel):
    root: Path = Path("data")
    checkpoint_every_n_updates: int = Field(default=10, ge=1)
    lock_timeout_seconds: float = Field(default=120, gt=0)


class TriadConfig(ConfigModel):
    student: StudentConfig = Field(default_factory=StudentConfig)
    lora: LoraSettings = Field(default_factory=LoraSettings)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    replay: ReplayConfig = Field(default_factory=ReplayConfig)
    teacher: TeacherConfig = Field(default_factory=TeacherConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)


def load_config(path: str | Path | None = None) -> TriadConfig:
    if path is None:
        return TriadConfig()
    with Path(path).open(encoding="utf-8") as handle:
        return TriadConfig.model_validate(yaml.safe_load(handle) or {})
