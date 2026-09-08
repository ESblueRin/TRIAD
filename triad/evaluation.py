"""Memory-free base/current/historical probes and teacher-forced distribution drift."""

import math
from difflib import SequenceMatcher

import torch
import torch.nn.functional as F

from .engine import TriadEngine
from .schemas import EvalCase


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def perplexity(loss: float) -> float | None:
    # Overflow is reported as null plus the finite loss, never invalid JSON Infinity.
    return math.exp(loss) if loss < 700 else None


def _distribution(student, encoded) -> torch.Tensor:
    with torch.inference_mode():
        logits = student.model(
            input_ids=encoded.input_ids.to(student.device),
            attention_mask=encoded.attention_mask.to(student.device),
            use_cache=False,
        ).logits[0, :-1]
        mask = encoded.labels[0, 1:] != -100
        logits = logits[mask.to(logits.device)].float()
        log_probs = F.log_softmax(logits, dim=-1).cpu()
    if not torch.isfinite(log_probs).all():
        raise FloatingPointError("Nonfinite evaluation distribution")
    return log_probs


def _score(log_probs, labels) -> dict:
    token_logp = log_probs.gather(1, labels[:, None]).squeeze(1)
    loss = float(-token_logp.mean())
    return {
        "target_log_probability": float(token_logp.sum()),
        "token_loss": loss,
        "target_tokens": len(labels),
        "perplexity": perplexity(loss),
    }


def _aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {}
    tokens = sum(row["target_tokens"] for row in rows)
    loss = -sum(row["target_log_probability"] for row in rows) / tokens
    base_loss = -sum(row["base_target_log_probability"] for row in rows) / tokens
    ppl, base_ppl = perplexity(loss), perplexity(base_loss)
    return {
        "cases": len(rows),
        "target_tokens": tokens,
        "token_loss": loss,
        "perplexity": ppl,
        "base_perplexity": base_ppl,
        "perplexity_change_from_base": ppl - base_ppl
        if ppl is not None and base_ppl is not None
        else None,
        "token_loss_change_from_base": loss - base_loss,
        "kl_base_to_adapter": sum(row["kl_base_to_adapter"] * row["target_tokens"] for row in rows)
        / tokens,
        "exact_match": sum(row["exact_match"] for row in rows) / len(rows),
        "response_similarity_to_base": sum(row["response_similarity_to_base"] for row in rows)
        / len(rows),
    }


def evaluate(
    engine: TriadEngine,
    user_id: str,
    cases: list[EvalCase],
    versions: list[int] | None = None,
    include_recall: bool = False,
) -> dict:
    if not cases and not include_recall:
        raise ValueError("Provide evaluation cases or enable learned interaction recall")
    store = engine.store(user_id)
    student = engine.student
    with engine.mutex, store.lock:
        state, current_version = engine._activate(store)
        original = student.adapter_state()
        cases = list(cases)
        if include_recall:
            # Explicitly label this as in-sample Teacher-target recall, not independent truth.
            cases += [
                EvalCase(prompt=entry.prompt[-1].content, target=entry.target, category="recall")
                for entry in state.replay
            ]
        if not cases:
            raise ValueError("No stored recall examples; provide an evaluation dataset")
        variants = [("base", None), ("current", current_version)]
        variants += [(f"checkpoint_{version}", version) for version in (versions or [])]
        report = {
            "user_id": user_id,
            "current_version": current_version,
            "metadata": engine.metadata(),
            "method": "No conversation history. KL(base || adapter) at teacher-forced target positions, "
            "including EOS. Recall probes use bounded replay targets without prior context.",
            "variants": {},
        }
        try:
            for name, version in variants:
                if name == "base":
                    student.restore_adapter(student.initial_adapter)
                elif name == "current":
                    student.restore_adapter(original)
                else:
                    historical, path = store.read(version)
                    if historical.signature != student.signature:
                        raise ValueError("Evaluation checkpoint configuration is incompatible")
                    student.restore_adapter(student.read_adapter(path))
                metrics = student.parameter_metrics()
                rows = []
                for case in cases:
                    prompt = engine.messages([], case.prompt)
                    encoded = student.encoder.encode(prompt, case.target)
                    labels = encoded.labels[0, 1:]
                    labels = labels[labels != -100]
                    with student.model.disable_adapter():
                        base_probs = _distribution(student, encoded)
                        base_response = student.generate(prompt)
                    if name == "base":
                        adapted_probs, response = base_probs, base_response
                    else:
                        adapted_probs = _distribution(student, encoded)
                        response = student.generate(prompt)
                    base_score = _score(base_probs, labels)
                    score = _score(adapted_probs, labels)
                    # double precision reduces cancellation near the frozen base.
                    kl = (
                        (base_probs.double().exp() * (base_probs.double() - adapted_probs.double()))
                        .sum(-1)
                        .mean()
                        .item()
                    )
                    rows.append(
                        {
                            **case.model_dump(),
                            **score,
                            "truncated": encoded.truncated,
                            "base_target_log_probability": base_score["target_log_probability"],
                            "kl_base_to_adapter": max(0.0, kl),
                            "response": response,
                            "base_response": base_response,
                            "exact_match": normalized(response) == normalized(case.target),
                            "response_similarity_to_base": SequenceMatcher(
                                None,
                                response,
                                base_response,
                                autojunk=False,
                            ).ratio(),
                        }
                    )
                report["variants"][name] = {
                    "adapter_version": version,
                    **metrics,
                    "aggregate": _aggregate(rows),
                    "by_category": {
                        category: _aggregate([r for r in rows if r["category"] == category])
                        for category in sorted({row["category"] for row in rows})
                    },
                    "cases": rows,
                }
        finally:
            student.restore_adapter(original)
        return report
