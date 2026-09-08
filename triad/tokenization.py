"""Completion-only causal LM labels, with explicit prompt/target token boundaries."""

from dataclasses import dataclass

import torch

from .schemas import Message


@dataclass
class EncodedExample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_length: int
    target_tokens: int
    truncated: bool

    def tensors(self, device: torch.device) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, name).to(device)
            for name in ("input_ids", "attention_mask", "labels")
        }


class CompletionEncoder:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def prompt_ids(self, messages: list[Message]) -> list[int]:
        records = [message.model_dump() for message in messages]
        if getattr(self.tokenizer, "chat_template", None):
            ids = self.tokenizer.apply_chat_template(
                records,
                tokenize=True,
                add_generation_prompt=True,
            )
        else:
            text = "".join(f"{m.role.capitalize()}: {m.content}\n" for m in messages)
            text += "Assistant: "
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            bos = self.tokenizer.bos_token_id
            if bos is not None:
                ids = [bos] + ids
        if not ids:
            raise ValueError("Tokenizer produced no prompt tokens")
        return list(ids)

    def encode(self, messages: list[Message], target: str) -> EncodedExample:
        if not target.strip():
            raise ValueError("A nonempty training target is required")
        prompt = self.prompt_ids(messages)
        completion = self.tokenizer.encode(target, add_special_tokens=False)
        if not completion:
            raise ValueError("Tokenizer produced no target tokens")
        eos = self.tokenizer.eos_token_id
        if eos is not None and completion[-1] != eos:
            completion.append(eos)
        original_length = len(prompt) + len(completion)
        # Preserve at least one causal context token. Left-truncate prompt first;
        # exceptionally long targets are right-truncated and their final EOS retained.
        if len(completion) > self.max_length - 1:
            completion = completion[: self.max_length - 1]
            if eos is not None:
                completion[-1] = eos
        prompt = prompt[-(self.max_length - len(completion)) :]
        ids = torch.tensor([prompt + completion], dtype=torch.long)
        labels = torch.tensor([[-100] * len(prompt) + completion], dtype=torch.long)
        return EncodedExample(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            labels=labels,
            prompt_length=len(prompt),
            target_tokens=len(completion),
            truncated=original_length > self.max_length,
        )
