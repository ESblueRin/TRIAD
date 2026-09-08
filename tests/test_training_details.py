import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from triad.schemas import Message, TrainingExample
from triad.student import Student


def test_model_loss_is_target_only_shifted_cross_entropy(student):
    encoded = student.encoder.encode([Message(role="user", content="Prompt only")], "target")
    with torch.no_grad():
        result = student.model(**encoded.tensors(student.device))
    shifted_labels = encoded.labels[:, 1:].reshape(-1)
    manual = torch.nn.functional.cross_entropy(
        result.logits[:, :-1].reshape(-1, result.logits.shape[-1]),
        shifted_labels,
        ignore_index=-100,
    )
    assert float(result.loss) == pytest.approx(float(manual), rel=1e-6)


def test_finite_exploding_gradients_are_clipped_before_step(student, monkeypatch):
    params = [p for p in student.model.parameters() if p.requires_grad]
    handles = [p.register_hook(lambda grad: grad * 1e7) for p in params]
    original = torch.optim.AdamW.step
    seen = []

    def observe(optimizer, *args, **kwargs):
        norm = torch.stack([p.grad.norm() for p in params if p.grad is not None]).norm()
        seen.append(float(norm))
        return original(optimizer, *args, **kwargs)

    monkeypatch.setattr(torch.optim.AdamW, "step", observe)
    result = student.train_interaction(
        TrainingExample(
            event_id="clip",
            prompt=[Message(role="user", content="Question")],
            target="Answer",
            strength=0.8,
        ),
        [],
        seed=12,
    )
    for handle in handles:
        handle.remove()
    assert result.performed
    assert result.gradient_norm > student.config.training.max_grad_norm
    assert max(seen) <= student.config.training.max_grad_norm * 1.00001


def test_gpt2_conv1d_architecture_is_inspected_and_trained(student, config):
    base = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(student.tokenizer),
            n_positions=256,
            n_embd=32,
            n_layer=1,
            n_head=4,
            bos_token_id=1,
            eos_token_id=2,
        )
    )
    gpt2 = Student(base, student.tokenizer, config)
    assert gpt2.target_modules == ["transformer.h.0.attn.c_attn", "transformer.h.0.attn.c_proj"]
    assert gpt2.model.peft_config["default"].fan_in_fan_out is True
    result = gpt2.train_interaction(
        TrainingExample(
            event_id="gpt2",
            prompt=[Message(role="user", content="q")],
            target="a",
            strength=0.9,
        ),
        [],
        seed=42,
    )
    assert result.performed, result.error
