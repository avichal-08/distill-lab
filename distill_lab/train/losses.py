import torch
import torch.nn.functional as F

IGNORE = -100


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


def fkl_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """forward kl, KL(teacher || student), per token over the full vocab.
    scaled by T^2 so its gradients stay comparable to ce as T changes (hinton et al.)"""
    mask = labels != IGNORE
    log_p = _log_probs(teacher_logits.detach(), temperature)
    log_q = _log_probs(student_logits, temperature)
    kl = (log_p.exp() * (log_p - log_q)).sum(-1)
    return _reduce(kl, mask, reduction) * temperature**2
