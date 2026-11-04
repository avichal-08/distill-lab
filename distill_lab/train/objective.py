import torch

from distill_lab.train.losses import IGNORE, TopK, ce_loss, fkl_loss, jsd_loss, rkl_loss, shift

KD = {"fkl": fkl_loss, "rkl": rkl_loss, "jsd": jsd_loss}


def objective(
    loss_cfg: dict,
    logits: torch.Tensor,
    labels: torch.Tensor,
    eos_id: int,
    teacher: TopK | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """summed loss over the batch's label tokens, plus its parts for logging.
    the trainer divides by the label-token count of the whole accumulation window.

    ce:        plain next-token cross-entropy on the labels
    fkl/rkl/jsd: (1 - w) * kd + w * ce, with w = loss.ce_weight.
    teacher is (B, T-1, K), already lined up with the shifted labels."""
    # (B, T, ...) in, flat (N, ...) over graded tokens from here on
    lg, lb = shift(logits, labels)
    # keep graded positions only. prompt and padding rows would each cost a full-vocab fp32 softmax
    graded = lb != IGNORE
    lg, lb = lg[graded], lb[graded]
    if teacher is not None:
        teacher = TopK(ids=teacher.ids[graded], logprobs=teacher.logprobs[graded])
    if loss_cfg["kind"] == "ce":
        ce = ce_loss(lg, lb, reduction="sum")
        return ce, {"ce": ce.item()}

    kd_fn = KD[loss_cfg["kind"]]
    w = loss_cfg["ce_weight"]
    is_eos = lb == eos_id
    # the teacher ends with <|im_end|>, the student with <|endoftext|>, and the cached row at that
    # position mostly predicts a newline anyway. so the stop token is graded by ce, never by the teacher
    kd = kd_fn(lg, teacher, lb.masked_fill(is_eos, IGNORE), loss_cfg["temperature"], reduction="sum")
    stop = ce_loss(lg, lb.masked_fill(~is_eos, IGNORE), reduction="sum")
    ce = ce_loss(lg, lb, reduction="sum")
    total = (1 - w) * (kd + stop) + w * ce
    return total, {"kd": kd.item(), "stop_ce": stop.item(), "ce": ce.item()}
