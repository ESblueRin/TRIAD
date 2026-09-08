"""Online orchestration: route -> train -> persist -> answer the follow-up."""

import hashlib
import importlib.metadata
import platform
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import TriadConfig
from .replay import append_replay, sample_replay
from .schemas import (
    Interaction,
    Message,
    PendingTurn,
    TrainingEvent,
    TrainingExample,
    TrainingResult,
    UserState,
)
from .storage import UserStore
from .student import Student
from .teacher import OllamaTeacher, Teacher, TeacherRouter


@dataclass
class ChatResult:
    response: str
    event: TrainingEvent | None
    adapter_version: int
    generation_error: str | None = None


class TriadEngine:
    def __init__(self, config: TriadConfig, student: Student, teacher: Teacher | None = None):
        self.config = config
        self.student = student
        self.router = TeacherRouter(teacher or OllamaTeacher(config.teacher), config.teacher)
        # A Student has one in-memory adapter slot. Serialize all uses, including evaluation.
        self.mutex = threading.RLock()
        self._stores: dict[str, UserStore] = {}
        self.runtime = {"python": platform.python_version(), "platform": platform.platform()}
        for package in ("torch", "transformers", "peft", "pydantic", "safetensors"):
            self.runtime[package] = importlib.metadata.version(package)

    @classmethod
    def from_config(cls, config: TriadConfig):
        return cls(config, Student.from_config(config))

    def store(self, user_id: str) -> UserStore:
        with self.mutex:
            if user_id not in self._stores:
                self._stores[user_id] = UserStore(self.config.storage, user_id)
            return self._stores[user_id]

    def metadata(self) -> dict:
        return {
            "runtime": self.runtime,
            "model": self.student.base_metadata,
            "config": self.config.model_dump(mode="json"),
        }

    def _activate(self, store: UserStore) -> tuple[UserState, int]:
        latest = store.latest()
        if latest is None:
            self.student.restore_adapter(self.student.initial_adapter)
            state = UserState(user_id=store.user_id, signature=self.student.signature)
            version, _ = store.commit(state, self.student.write_adapter, self.metadata(), pin=True)
            return state, version
        state, path, version = latest
        if state.signature != self.student.signature:
            raise ValueError(
                "Stored adapter has a different base/tokenizer/LoRA fingerprint. "
                "Use the original model configuration or a new storage root."
            )
        self.student.restore_adapter(self.student.read_adapter(path))
        return state, version

    def _context(self, history: list[Message]) -> list[Message]:
        limit = self.config.student.context_messages
        context = history[-limit:] if limit else []
        # Do not start a truncated chat with an orphaned assistant response.
        if context and context[0].role == "assistant":
            context = context[1:]
        return context

    def messages(self, context: list[Message], query: str) -> list[Message]:
        return [
            Message(role="system", content=self.config.student.system_prompt),
            *context,
            Message(role="user", content=query),
        ]

    def _learn(
        self,
        store: UserStore,
        state: UserState,
        interaction: Interaction,
    ) -> tuple[UserState, TrainingEvent, int]:
        before = self.student.adapter_state()
        state = state.model_copy(deep=True)
        try:
            event_id = uuid.uuid4().hex
            routed = self.router.route(interaction)
            decision = routed.decision
            strength = max(decision.update_strength, self.config.training.minimum_update_strength)
            # Reproducible across restart/rollback, independent of UUIDs and wall-clock time.
            seed_bytes = f"{self.config.student.seed}:{state.user_id}:{state.interactions}".encode()
            seed = int.from_bytes(hashlib.sha256(seed_bytes).digest()[:4], "big")
            sampled = sample_replay(state.replay, self.config.replay, seed)
            if decision.should_update:
                example = TrainingExample(
                    event_id=event_id,
                    prompt=self.messages(interaction.context, interaction.user_query),
                    target=decision.training_target,
                    strength=strength,
                )
                result = self.student.train_interaction(example, sampled, seed)
                if result.performed:
                    state.updates += 1
                    state.replay = append_replay(state.replay, example, self.config.replay)
            else:
                sampled = []
                result = TrainingResult(performed=False, error=decision.technical_failure)
            state.interactions += 1
            state.last_event_id = event_id
            state.pending = None  # The follow-up is consumed exactly once, even after restart.
            event = TrainingEvent(
                event_id=event_id,
                user_id=state.user_id,
                user_query=interaction.user_query,
                student_response=interaction.student_response,
                user_followup=interaction.user_followup,
                context=interaction.context,
                teacher_feedback_type=decision.feedback_type,
                teacher_interpretation=decision.interpretation,
                teacher_target=decision.training_target,
                teacher_confidence=decision.confidence,
                teacher_decision=routed,
                update_strength=decision.update_strength,
                effective_update_strength=strength * self.config.training.update_strength_scale,
                training_performed=result.performed,
                loss=result.loss,
                gradient_norm=result.gradient_norm,
                training=result,
                replay_event_ids=[item.event_id for item in sampled],
                config=self.config.model_dump(mode="json"),
            )
            version, _ = store.commit(
                state, self.student.write_adapter, self.metadata(), event=event
            )
            return state, event, version
        except BaseException:
            self.student.restore_adapter(before)
            raise

    def chat(self, user_id: str, text: str) -> ChatResult:
        if not text.strip():
            raise ValueError("Enter a nonempty message")
        store = self.store(user_id)
        with self.mutex, store.lock:
            state, version = self._activate(store)
            event = None
            if state.pending is not None:
                pending = state.pending
                state, event, version = self._learn(
                    store,
                    state,
                    Interaction(
                        user_query=pending.query,
                        student_response=pending.response,
                        user_followup=text,
                        context=pending.context,
                    ),
                )
            context = self._context(state.history)
            try:
                response = self.student.generate(self.messages(context, text))
            except (RuntimeError, ValueError, MemoryError) as exc:
                # Any preceding update is already durable. Do not undo it or train it twice.
                error = f"{type(exc).__name__}: {str(exc)[:400]}"
                version, _ = store.commit(
                    state,
                    self.student.write_adapter,
                    self.metadata(),
                    note={
                        "kind": "generation_failure",
                        "user_query": text,
                        "error": error,
                    },
                )
                return ChatResult("", event, version, error)
            state.pending = PendingTurn(query=text, response=response, context=context)
            state.history = self._context(
                context
                + [
                    Message(role="user", content=text),
                    Message(role="assistant", content=response),
                ]
            )
            version, _ = store.commit(
                state,
                self.student.write_adapter,
                self.metadata(),
                note={
                    "kind": "conversation",
                    "user_query": text,
                    "student_response": response,
                },
            )
            return ChatResult(response, event, version)

    def learn(self, user_id: str, interaction: Interaction) -> TrainingEvent:
        """Explicit triple API for controlled datasets; no prompt history is injected."""
        store = self.store(user_id)
        with self.mutex, store.lock:
            state, _ = self._activate(store)
            _, event, _ = self._learn(store, state, interaction)
            return event

    def answer(self, user_id: str, query: str) -> str:
        """Stateless probe: uses current weights, never conversation memory or training."""
        store = self.store(user_id)
        with self.mutex, store.lock:
            self._activate(store)
            return self.student.generate(self.messages([], query))

    def checkpoint(self, user_id: str) -> tuple[int, Path]:
        store = self.store(user_id)
        with self.mutex, store.lock:
            state, _ = self._activate(store)
            return store.commit(
                state,
                self.student.write_adapter,
                self.metadata(),
                pin=True,
                note={"kind": "manual_checkpoint"},
            )

    def rollback(self, user_id: str, version: int) -> int:
        store = self.store(user_id)
        with self.mutex, store.lock:
            self._activate(store)
            before = self.student.adapter_state()
            historical, path = store.read(version)
            if historical.signature != self.student.signature:
                raise ValueError("Checkpoint configuration is incompatible")
            try:
                self.student.restore_adapter(self.student.read_adapter(path))
                restored, _ = store.commit(
                    historical,
                    self.student.write_adapter,
                    self.metadata(),
                    pin=True,
                    note={"kind": "rollback", "source_version": version},
                )
                return restored
            except BaseException:
                self.student.restore_adapter(before)
                raise

    def new_conversation(self, user_id: str) -> int:
        store = self.store(user_id)
        with self.mutex, store.lock:
            state, _ = self._activate(store)
            state.pending = None
            state.history = []
            version, _ = store.commit(
                state,
                self.student.write_adapter,
                self.metadata(),
                note={"kind": "new_conversation"},
            )
            return version
