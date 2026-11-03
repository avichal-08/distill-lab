import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F

IGNORE = -100
# neither tail goes below this. a zero teacher tail makes reverse kl infinite,
# a zero student tail does the same to forward kl
TAIL_FLOOR = 1e-6


@dataclass
class TopK:
    """cached teacher distribution: the k most likely ids per position and their
    log-probs from the full softmax at T=1, both (B, T, K)"""

    ids: torch.Tensor
    logprobs: torch.Tensor


def shift(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # logits at position t predict the token at t+1
    return logits[:, :-1], labels[:, 1:]


def _reduce(per_token: torch.Tensor, mask: torch.Tensor, reduction: str) -> torch.Tensor:
    total = (per_token * mask).sum()
    if reduction == "sum":
        return total
    if reduction == "mean":
        return total / mask.sum().clamp_min(1)
    raise ValueError(f"unknown reduction {reduction!r}")


def ce_loss(logits: torch.Tensor, labels: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """next-token ce on shifted (B, T, V) logits, IGNORE positions skipped.
    use reduction="sum" under grad accumulation and divide by the full batch's token count"""
    mask = labels != IGNORE
    # fp32 here, fp16 log_softmax over 150k entries loses the small probs
    logp = F.log_softmax(logits.float(), dim=-1)
    nll = -logp.gather(-1, labels.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    return _reduce(nll, mask, reduction)


def _log_probs(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    logp = F.log_softmax(logits.float() / temperature, dim=-1)
    # -inf would turn 0 * log(0) into nan, the smallest finite float keeps it at 0
    return logp.clamp_min(torch.finfo(logp.dtype).min)


def _topk_buckets(student_logits: torch.Tensor, teacher: TopK) -> tuple[torch.Tensor, torch.Tensor]:
    """collapse both distributions to k+1 buckets: the teacher's top-k ids plus one
    "everything else" bucket. returns (B, T, K+1) log-probs for teacher and student"""
    log_p_k = teacher.logprobs.detach().float()
    # fp16 cache rounding can push the top-k sum slightly past 1
    tail = (1 - log_p_k.exp().sum(-1, keepdim=True)).clamp_min(TAIL_FLOOR)
    log_p = torch.cat([log_p_k, tail.log()], dim=-1)

    log_q_full = F.log_softmax(student_logits.float(), dim=-1)
    log_q_k = log_q_full.gather(-1, teacher.ids)
    # exact tail: logsumexp over everything that isn't in the top-k, not 1 - sum
    log_q_tail = log_q_full.scatter(-1, teacher.ids, float("-inf")).logsumexp(-1, keepdim=True)
    log_q_tail = log_q_tail.clamp_min(math.log(TAIL_FLOOR))
    log_q = torch.cat([log_q_k, log_q_tail], dim=-1)
    return log_p, log_q


def _kl(log_a: torch.Tensor, log_b: torch.Tensor) -> torch.Tensor:
    return (log_a.exp() * (log_a - log_b)).sum(-1)


def _distill(
    per_token_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    student_logits: torch.Tensor,
    teacher: torch.Tensor | TopK,
    labels: torch.Tensor,
    temperature: float,
    reduction: str,
) -> torch.Tensor:
    if isinstance(teacher, TopK):
        # buckets come out at T=1, so temper them as a k+1-way distribution
        log_p, log_q = _topk_buckets(student_logits, teacher)
        log_p, log_q = _log_probs(log_p, temperature), _log_probs(log_q, temperature)
    else:
        log_p = _log_probs(teacher.detach(), temperature)
        log_q = _log_probs(student_logits, temperature)
    per_token = per_token_fn(log_p, log_q)
    # T^2 keeps gradients comparable to ce as T changes (hinton et al.)
    return _reduce(per_token, labels != IGNORE, reduction) * temperature**2


def fkl_loss(
    student_logits: torch.Tensor,
    teacher: torch.Tensor | TopK,
    labels: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """forward kl, KL(teacher || student). mass-covering"""
    return _distill(lambda p, q: _kl(p, q), student_logits, teacher, labels, temperature, reduction)


def rkl_loss(
    student_logits: torch.Tensor,
    teacher: torch.Tensor | TopK,
    labels: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """reverse kl, KL(student || teacher). mode-seeking"""
    # infinite wherever the teacher has exactly zero mass, b4's tail bucket avoids that
    return _distill(lambda p, q: _kl(q, p), student_logits, teacher, labels, temperature, reduction)


def jsd_loss(
    student_logits: torch.Tensor,
    teacher: torch.Tensor | TopK,
    labels: torch.Tensor,
    temperature: float = 1.0,
    beta: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """generalized jsd from gkd (agarwal et al.): beta*KL(p||m) + (1-beta)*KL(q||m),
    m = beta*p + (1-beta)*q. beta=0.5 is the usual symmetric one, bounded by ln 2"""

    def per_token(log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
        log_m = torch.logsumexp(torch.stack([log_p + math.log(beta), log_q + math.log(1 - beta)]), dim=0)
        return beta * _kl(log_p, log_m) + (1 - beta) * _kl(log_q, log_m)

    return _distill(per_token, student_logits, teacher, labels, temperature, reduction)
