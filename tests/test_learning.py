import pytest
import torch

from triad.engine import TriadEngine
from triad.schemas import Interaction, Message, TrainingExample
from triad.student import delta_norm, resolve_targets
from triad.tiny import make_tiny_student


def interaction(followup="아니야. 내 이름은 주호야."):
    return Interaction(
        user_query="내 이름이 뭐야?", student_response="성준입니다.", user_followup=followup
    )


@pytest.mark.parametrize(
    "kind,followup,target,confidence",
    [
        ("corrective", "아니야. 내 이름은 주호야.", "주호입니다.", 0.99),
        ("hostile", "ㅅㅂ 진짜 답답하네.", "정확히 기억하지 못하겠어.", 0.01),
        ("noise", "ㅋㅋㅋㅋㅋㅋ", "다시 생각해서 답할게.", 0.0),
        ("ambiguous", "아무튼 아니야", "잘못 답한 것 같아.", 0.0),
        ("positive", "맞아 고마워", "성준입니다.", 0.8),
        ("corrective", "내 이름은 바나나야.", "바나나입니다.", 0.9),
    ],
)
def test_all_feedback_changes_only_lora(engine, teacher, kind, followup, target, confidence):
    teacher.decision = teacher.decision.model_copy(
        update={
            "feedback_type": kind,
            "training_target": target,
            "confidence": confidence,
        }
    )
    student = engine.student
    base = {n: p.detach().clone() for n, p in student.model.named_parameters() if "lora_" not in n}
    before = student.adapter_state()
    event = engine.learn("juho", interaction(followup))
    assert event.training_performed, event.training.error
    assert event.teacher_target == target
    assert event.training.parameter_delta_norm > 0
    assert delta_norm(student.adapter_state(), before) > 0
    for name, p in student.model.named_parameters():
        if name in base:
            assert not p.requires_grad
            assert p.grad is None
            assert torch.equal(p, base[name])
    assert engine.store("juho").events()[0]["training_performed"]


def test_only_target_labels_including_shift_and_eos(student):
    messages = [Message(role="system", content="RULE"), Message(role="user", content="QUESTION")]
    encoded = student.encoder.encode(messages, "ANSWER")
    boundary = encoded.prompt_length
    assert torch.all(encoded.labels[0, :boundary] == -100)
    expected = student.tokenizer.encode("ANSWER", add_special_tokens=False) + [
        student.tokenizer.eos_token_id
    ]
    assert encoded.labels[0, boundary:].tolist() == expected
    assert (encoded.labels[0, 1:] != -100).sum() == len(expected)
    assert encoded.input_ids[0, boundary:].tolist() == expected
    assert encoded.attention_mask.all()


def test_truncation_preserves_target_and_context(student):
    encoded = student.encoder.encode([Message(role="user", content="long " * 1000)], "target")
    assert encoded.truncated
    assert encoded.input_ids.shape[1] == student.config.student.max_length
    assert encoded.prompt_length >= 1
    assert encoded.target_tokens == 7  # six byte tokens and EOS
    very_long = student.encoder.encode([Message(role="user", content="q")], "x" * 1000)
    assert very_long.prompt_length == 1
    assert very_long.labels[0, -1].item() == student.tokenizer.eos_token_id


@pytest.mark.parametrize("optimizer", ["adamw", "sgd"])
def test_strength_scales_parameter_update(student, config, optimizer):
    config.training.optimizer = optimizer
    initial = student.adapter_state()
    prompt = [Message(role="user", content="What is my name?")]
    small = student.train_interaction(
        TrainingExample(
            event_id="small",
            prompt=prompt,
            target="Juho",
            strength=0.1,
        ),
        [],
        seed=42,
    )
    student.restore_adapter(initial)
    large = student.train_interaction(
        TrainingExample(
            event_id="large",
            prompt=prompt,
            target="Juho",
            strength=0.9,
        ),
        [],
        seed=42,
    )
    assert small.performed and large.performed
    assert large.effective_learning_rate == pytest.approx(small.effective_learning_rate * 9)
    assert large.parameter_delta_norm > small.parameter_delta_norm * 8


@pytest.mark.parametrize("enabled", [True, False])
def test_replay_on_off(engine, config, enabled):
    config.replay.enabled = enabled
    config.replay.capacity = 2
    events = [engine.learn("replay", interaction()) for _ in range(4)]
    assert all(e.training_performed for e in events)
    assert events[-1].training.replay_count == (2 if enabled else 0)
    state, _, _ = engine.store("replay").latest()
    assert len(state.replay) == 2
    assert bool(events[-1].replay_event_ids) is enabled


def test_new_users_start_identically_and_are_isolated(engine):
    engine.checkpoint("B")
    b_before = engine.student.adapter_state()
    b_versions = engine.store("B").checkpoints()
    engine.learn("A", interaction())
    a_state = engine.student.adapter_state()
    assert delta_norm(a_state, b_before) > 0
    engine.answer("B", "My name?")
    assert delta_norm(engine.student.adapter_state(), b_before) == 0
    assert engine.store("B").checkpoints() == b_versions
    assert engine.store("B").events() == []
    engine.answer("new-user", "My name?")
    assert delta_norm(engine.student.adapter_state(), b_before) == 0


def test_restart_restores_parameters_and_replay(engine, config, teacher):
    engine.learn("persist", interaction())
    before = engine.student.adapter_state()
    new_engine = TriadEngine(config, make_tiny_student(config), teacher)
    new_engine.answer("persist", "My name?")
    assert delta_norm(new_engine.student.adapter_state(), before) == 0
    state, _, _ = new_engine.store("persist").latest()
    assert state.updates == 1 and len(state.replay) == 1


def test_target_resolution_rejects_missing_and_output_head(student):
    base = make_tiny_student(student.config).model.get_base_model()
    with pytest.raises(ValueError, match="matches no"):
        resolve_targets(base, ["does_not_exist"])
    with pytest.raises(ValueError, match="output head"):
        resolve_targets(base, ["lm_head"])
    assert len(student.target_modules) == 8
    assert all("self_attn" in name for name in student.target_modules)


def test_zero_teacher_strength_still_learns_by_default(engine, teacher):
    teacher.decision = teacher.decision.model_copy(
        update={"update_strength": 0.0, "confidence": 0.0}
    )
    event = engine.learn("zero", interaction("ㅋㅋㅋㅋ"))
    assert event.training_performed
    assert event.effective_update_strength == 0.05
