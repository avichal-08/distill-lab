import pytest
import torch

from distill_lab.train.losses import IGNORE, TopK, ce_loss, fkl_loss, shift
from distill_lab.train.objective import objective

EOS = 9
V = 12


def batch():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, V)
    labels = torch.tensor([[IGNORE, 3, 4, 5, EOS], [IGNORE, IGNORE, 6, EOS, IGNORE]])
    # teacher rows on the shifted positions, random but valid
    lp, ids = torch.randn(2, 4, V).log_softmax(-1).topk(4, dim=-1)
    return logits, labels, TopK(ids=ids, logprobs=lp)


def cfg(kind, w=0.1):
    return {"kind": kind, "ce_weight": w, "temperature": 1.0}


def test_ce_is_plain_summed_cross_entropy():
    logits, labels, _ = batch()
    total, parts = objective(cfg("ce"), logits, labels, EOS)
    assert total.item() == pytest.approx(ce_loss(*shift(logits, labels), reduction="sum").item(), rel=1e-6)
    assert set(parts) == {"ce"}


def test_kd_mixes_with_ce_by_the_weight():
    logits, labels, teacher = batch()
    total, p = objective(cfg("fkl", w=0.25), logits, labels, EOS, teacher)
    assert total.item() == pytest.approx(0.75 * (p["kd"] + p["stop_ce"]) + 0.25 * p["ce"], rel=1e-5)


def test_stop_token_is_never_graded_by_the_teacher():
    logits, labels, teacher = batch()
    _, base = objective(cfg("fkl"), logits, labels, EOS, teacher)
    # scramble the teacher only at the positions that predict eos
    lg, lb = shift(logits, labels)
    eos_pos = lb == EOS
    teacher.logprobs[eos_pos] = torch.randn(int(eos_pos.sum()), V).log_softmax(-1)[:, :4]
    _, changed = objective(cfg("fkl"), logits, labels, EOS, teacher)
    assert changed["kd"] == pytest.approx(base["kd"])
    # and the stop part is exactly ce on those positions
    want = ce_loss(lg, lb.masked_fill(~eos_pos, IGNORE), reduction="sum").item()
    assert base["stop_ce"] == pytest.approx(want, rel=1e-6)


def test_kd_part_matches_the_loss_function_directly():
    logits, labels, teacher = batch()
    _, p = objective(cfg("fkl"), logits, labels, EOS, teacher)
    lg, lb = shift(logits, labels)
    want = fkl_loss(lg, teacher, lb.masked_fill(lb == EOS, IGNORE), reduction="sum").item()
    assert p["kd"] == pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("kind", ["fkl", "rkl", "jsd"])
def test_every_kd_kind_backprops(kind):
    logits, labels, teacher = batch()
    logits.requires_grad_()
    total, _ = objective(cfg(kind), logits, labels, EOS, teacher)
    total.backward()
    assert torch.isfinite(logits.grad).all() and logits.grad.abs().sum() > 0
