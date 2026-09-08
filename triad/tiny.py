"""Download-free REAL Transformer/PEFT backend for tests and a mechanical demo.

Random weights and byte-level vocabulary: not a language-capable assistant.
"""

import torch
from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from .config import TriadConfig
from .schemas import Interaction, TeacherDecision
from .student import Student


def tiny_config(root="data/tiny-demo") -> TriadConfig:
    return TriadConfig.model_validate(
        {
            "student": {
                "model_name": "triad-tiny-random",
                "device": "cpu",
                "max_length": 256,
                "max_new_tokens": 12,
                "context_messages": 4,
            },
            "lora": {"rank": 4, "alpha": 8, "dropout": 0.0},
            "storage": {"root": root, "checkpoint_every_n_updates": 2},
        }
    )


def make_tiny_student(config: TriadConfig) -> Student:
    vocab = {token: i for i, token in enumerate(["<pad>", "<bos>", "<eos>", "<unk>"])}
    vocab.update(
        {char: i + 4 for i, char in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    )
    backend = Tokenizer(models.BPE(vocab=vocab, merges=[], unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        bos_token="<bos>",
        eos_token="<eos>",
        unk_token="<unk>",
    )
    tokenizer.chat_template = (
        "{{ bos_token }}{% for message in messages %}"
        "{{ message['role'] + ': ' + message['content'] + '\n' }}{% endfor %}"
        "{% if add_generation_prompt %}{{ 'assistant: ' }}{% endif %}"
    )
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(config.student.seed)
        base = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=len(vocab),
                hidden_size=32,
                intermediate_size=64,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=config.student.max_length,
                bos_token_id=1,
                eos_token_id=2,
                pad_token_id=0,
            )
        )
    return Student(base, tokenizer, config)


class DemoTeacher:
    """Scripted targets only for mechanical offline verification, never the default Teacher."""

    def decide(self, interaction: Interaction) -> TeacherDecision:
        text = interaction.user_followup
        if "이름" in text:
            target = "바나나입니다." if "바나나" in text else "주호입니다."
            kind, strength = "corrective", 0.9
        elif any(word in text for word in ("ㅅㅂ", "멍청", "못하냐", "답답")):
            target, kind, strength = "정확히 기억하지 못하겠어.", "hostile", 0.2
        else:
            target, kind, strength = "다시 생각해서 답할게.", "ambiguous", 0.15
        return TeacherDecision(
            feedback_type=kind,
            confidence=0.5,
            interpretation="Scripted offline demo fixture; not an Ollama judgment.",
            training_target=target,
            update_strength=strength,
        )
