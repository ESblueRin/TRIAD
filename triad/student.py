"""Inspectable PEFT training with an immutable base and one replaceable adapter slot."""

import hashlib
import json
import math
from contextlib import contextmanager
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from .config import TriadConfig
from .schemas import Message, TrainingExample, TrainingResult
from .tokenization import CompletionEncoder


def tensor_digest(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(tensors.items()):
        value = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((list(value.shape), value.dtype)).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_norm(tensors: dict[str, torch.Tensor]) -> float:
    return math.sqrt(math.fsum(t.double().square().sum().item() for t in tensors.values()))


def delta_norm(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    return tensor_norm(
        {name: value.double() - right[name].double() for name, value in left.items()}
    )


def resolve_targets(model: nn.Module, requested: list[str] | str) -> list[str]:
    """Resolve full paths before PEFT injection; never silently accept missing suffixes."""
    linear = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, (nn.Linear, Conv1D))
    }
    output_layer = model.get_output_embeddings()
    if requested == "auto":
        families = [
            ("q_proj", "k_proj", "v_proj", "o_proj"),
            ("c_attn", "c_proj"),
            ("query_key_value", "dense"),
        ]
        selected = []
        parents = {name.rsplit(".", 1)[0] for name in linear if "." in name}
        for parent in sorted(parents):
            if not any(word in parent.lower() for word in ("attn", "attention")):
                continue
            for family in families:
                names = [f"{parent}.{leaf}" for leaf in family]
                if all(name in linear for name in names):
                    selected.extend(names)
                    break
        if not selected:
            candidates = ", ".join(list(linear)[:40])
            raise ValueError(
                "Cannot safely infer attention projections. Set lora.target_modules "
                f"explicitly. Available linear paths: {candidates}"
            )
    else:
        selected = []
        for suffix in requested:
            matches = [name for name in linear if name == suffix or name.endswith("." + suffix)]
            if not matches:
                raise ValueError(f"LoRA target {suffix!r} matches no Linear/Conv1D module")
            selected.extend(matches)
    selected = sorted(set(selected))
    if any(linear[name] is output_layer for name in selected):
        raise ValueError("The output head is protected; select Transformer projections")
    if len({type(linear[name]) for name in selected}) > 1:
        raise ValueError("Mixed Linear/Conv1D targets require separate architecture support")
    return selected


class Student:
    def __init__(self, base_model, tokenizer, config: TriadConfig):
        self.config = config
        self.tokenizer = tokenizer
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer needs a pad or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        requested_device = config.student.device
        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if requested_device == "auto"
            else torch.device(requested_device)
        )
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested, but this PyTorch build has no available CUDA device")
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        dtype = getattr(torch, config.student.dtype)
        base_model.to(device=self.device, dtype=dtype)
        base_model.requires_grad_(False)
        self.base_digest = tensor_digest(dict(base_model.named_parameters()))
        self.target_modules = resolve_targets(base_model, config.lora.target_modules)
        module_map = dict(base_model.named_modules())
        fan_in_fan_out = isinstance(module_map[self.target_modules[0]], Conv1D)
        self.base_metadata = {
            "model_name": config.student.model_name,
            "revision": config.student.revision,
            "resolved_revision": getattr(base_model.config, "_commit_hash", None),
            "model_type": base_model.config.model_type,
            "model_config": base_model.config.to_dict(),
            "base_sha256": self.base_digest,
            "dtype": config.student.dtype,
            "seed": config.student.seed,
            "lora": config.lora.model_dump(),
            "resolved_target_modules": self.target_modules,
            "tokenizer_sha256": hashlib.sha256(
                json.dumps(
                    {
                        "vocab": tokenizer.get_vocab(),
                        "chat_template": getattr(tokenizer, "chat_template", None),
                        "special_tokens": tokenizer.special_tokens_map,
                    },
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode()
            ).hexdigest(),
        }
        with self.seeded(config.student.seed):
            self.model = get_peft_model(
                base_model,
                LoraConfig(
                    r=config.lora.rank,
                    lora_alpha=config.lora.alpha,
                    lora_dropout=config.lora.dropout,
                    target_modules=self.target_modules,
                    task_type="CAUSAL_LM",
                    bias="none",
                    fan_in_fan_out=fan_in_fan_out,
                ),
            )
        injected = {
            name
            for name, module in self.model.get_base_model().named_modules()
            if hasattr(module, "lora_A") and hasattr(module, "lora_B")
        }
        if injected != set(self.target_modules):
            raise RuntimeError("PEFT injection paths differ from the inspected target modules")
        # FP32 LoRA prevents tiny online updates vanishing in reduced-precision weights.
        for name, param in self.model.named_parameters():
            if "lora_" in name:
                param.data = param.data.float()
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        self.assert_isolation()
        if config.student.gradient_checkpointing:
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        self.encoder = CompletionEncoder(tokenizer, config.student.max_length)
        self.initial_adapter = self.adapter_state()
        self.base_metadata["initial_adapter_sha256"] = tensor_digest(self.initial_adapter)
        self.signature = hashlib.sha256(
            json.dumps(self.base_metadata, sort_keys=True).encode()
        ).hexdigest()
        self.model.eval()

    @classmethod
    def from_config(cls, config: TriadConfig) -> "Student":
        options = {
            "revision": config.student.revision,
            "local_files_only": config.student.local_files_only,
            "trust_remote_code": False,
        }
        tokenizer = AutoTokenizer.from_pretrained(config.student.model_name, **options)
        model = AutoModelForCausalLM.from_pretrained(
            config.student.model_name,
            torch_dtype=getattr(torch, config.student.dtype),
            **options,
        )
        return cls(model, tokenizer, config)

    @contextmanager
    def seeded(self, seed: int):
        devices = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.random.default_generator.manual_seed(seed)
            if devices:
                with torch.cuda.device(self.device):
                    torch.cuda.manual_seed(seed)
            yield

    def assert_isolation(self):
        trainable = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        if not trainable or any(".lora_A." not in n and ".lora_B." not in n for n, _ in trainable):
            raise RuntimeError("Only LoRA A/B parameters may be trainable")
        if any(p.grad is not None for n, p in self.model.named_parameters() if "lora_" not in n):
            raise RuntimeError("Frozen base unexpectedly has gradients")

    def adapter_state(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu().clone().contiguous()
            for name, value in get_peft_model_state_dict(
                self.model, save_embedding_layers=False
            ).items()
        }

    def validate_adapter(self, state: dict[str, torch.Tensor]):
        expected = self.initial_adapter
        if set(state) != set(expected):
            raise ValueError("Adapter tensor names do not match the current architecture")
        for name, tensor in state.items():
            if tensor.shape != expected[name].shape or tensor.dtype != expected[name].dtype:
                raise ValueError(f"Incompatible adapter tensor: {name}")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"Nonfinite adapter tensor: {name}")

    def restore_adapter(self, state: dict[str, torch.Tensor]):
        self.validate_adapter(state)
        set_peft_model_state_dict(self.model, state, adapter_name="default")
        self.model.zero_grad(set_to_none=True)
        self.model.eval()
        self.assert_isolation()

    def write_adapter(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        state = self.adapter_state()
        self.validate_adapter(state)
        save_file(state, str(path / "adapter_model.safetensors"), metadata={"format": "pt"})
        self.model.peft_config["default"].save_pretrained(path)

    def read_adapter(self, path: Path) -> dict[str, torch.Tensor]:
        state = load_file(str(path / "adapter_model.safetensors"), device="cpu")
        self.validate_adapter(state)
        return state

    def train_interaction(
        self,
        example: TrainingExample,
        replay: list[TrainingExample],
        seed: int,
    ) -> TrainingResult:
        self.assert_isolation()
        before = self.adapter_state()
        cfg = self.config.training
        effective_lr = cfg.learning_rate * cfg.update_strength_scale * example.strength
        losses, norms = [], []
        target_tokens = 0
        try:
            params = [p for p in self.model.parameters() if p.requires_grad]
            # Reset optimizer state per interaction: no shared momentum between users,
            # and replay/checkpoint restoration does not need hidden optimizer history.
            optimizer_cls = torch.optim.AdamW if cfg.optimizer == "adamw" else torch.optim.SGD
            optimizer = optimizer_cls(params, lr=effective_lr, weight_decay=0.0)
            current_encoded = self.encoder.encode(example.prompt, example.target)
            target_tokens = current_encoded.target_tokens
            encoded_replay = [(self.encoder.encode(e.prompt, e.target), e.strength) for e in replay]
            with self.seeded(seed):
                self.model.train()
                for _ in range(cfg.steps_per_interaction):
                    optimizer.zero_grad(set_to_none=True)
                    mix = self.config.replay.loss_weight if encoded_replay else 0.0
                    # Sequential microbatches bound activation memory. Each backward
                    # only sees that example's assistant target tokens.
                    batches = [(current_encoded, 1.0 - mix)] + [
                        (encoded, mix * strength / len(encoded_replay))
                        for encoded, strength in encoded_replay
                    ]
                    step_loss = 0.0
                    for encoded, weight in batches:
                        loss = self.model(**encoded.tensors(self.device), use_cache=False).loss
                        if not torch.isfinite(loss):
                            raise FloatingPointError("Nonfinite loss")
                        step_loss += float(loss.detach()) * weight
                        (loss * weight).backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        params,
                        cfg.max_grad_norm,
                        error_if_nonfinite=True,
                    )
                    norms.append(float(norm))
                    optimizer.step()
                    if any(not torch.isfinite(p).all() for p in params):
                        raise FloatingPointError("Optimizer produced nonfinite LoRA weights")
                    losses.append(step_loss)
            self.assert_isolation()
            change = delta_norm(self.adapter_state(), before)
            if not math.isfinite(change) or change == 0:
                raise FloatingPointError("No finite, representable LoRA change")
            return TrainingResult(
                performed=True,
                loss=losses[-1],
                losses=losses,
                gradient_norm=max(norms),
                parameter_delta_norm=change,
                effective_learning_rate=effective_lr,
                replay_count=len(replay),
                target_tokens=target_tokens,
                seed=seed,
            )
        except (RuntimeError, FloatingPointError, ValueError, MemoryError) as exc:
            error = f"{type(exc).__name__}: {str(exc)[:400]}"
            # Release optimizer tensors and failed activation graphs before GPU rollback.
            self.model.zero_grad(set_to_none=True)
            if "optimizer" in locals():
                optimizer.state.clear()
            if "loss" in locals():
                del loss
            exc.__traceback__ = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            self.restore_adapter(before)
            return TrainingResult(
                performed=False,
                effective_learning_rate=effective_lr,
                replay_count=len(replay),
                target_tokens=target_tokens,
                seed=seed,
                error=error,
            )
        except BaseException:
            self.restore_adapter(before)
            raise
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.eval()

    def generate(self, messages: list[Message]) -> str:
        self.model.eval()
        cfg = self.config.student
        prompt = self.encoder.prompt_ids(messages)[-(cfg.max_length - cfg.max_new_tokens) :]
        ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=cfg.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )
        return self.tokenizer.decode(output[0, len(prompt) :], skip_special_tokens=True).strip()

    def parameter_metrics(self) -> dict[str, float]:
        state = self.adapter_state()
        return {
            "lora_weight_norm": tensor_norm(state),
            "parameter_delta_from_initial": delta_norm(state, self.initial_adapter),
        }
