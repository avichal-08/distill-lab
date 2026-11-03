import math

import pytest
import torch
import torch.nn.functional as F

from distill_lab.train.losses import IGNORE, ce_loss, fkl_loss, shift


def test_shift_lines_up_prediction_with_next_token():
    logits = torch.arange(4).float().view(1, 4, 1)
    labels = torch.tensor([[10, 11, 12, 13]])
    lg, lb = shift(logits, labels)
    # position 0 predicts token 11
    assert lg[0, :, 0].tolist() == [0, 1, 2]
    assert lb[0].tolist() == [11, 12, 13]


def test_hand_computed_value():
    # q = [0.5, 0.3, 0.2], right token 0 -> -ln 0.5
    logits = torch.tensor([[[0.5, 0.3, 0.2]]]).log()
    labels = torch.tensor([[0]])
    assert ce_loss(logits, labels).item() == pytest.approx(math.log(2), abs=1e-6)


def test_gradient_is_q_minus_onehot():
    logits = torch.tensor([[[0.5, 0.3, 0.2]]]).log().requires_grad_()
    ce_loss(logits, torch.tensor([[0]])).backward()
    assert torch.allclose(logits.grad[0, 0], torch.tensor([-0.5, 0.3, 0.2]), atol=1e-6)


def test_matches_torch_with_ignored_positions():
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 7)
    labels = torch.randint(0, 7, (2, 5))
    labels[0, :2] = IGNORE
    labels[1, 4] = IGNORE
    ref = F.cross_entropy(logits.view(-1, 7), labels.view(-1), ignore_index=IGNORE)
    assert ce_loss(logits, labels).item() == pytest.approx(ref.item(), abs=1e-6)


def test_sum_over_microbatches_equals_full_batch_mean():
    torch.manual_seed(0)
    logits = torch.randn(4, 6, 9)
    labels = torch.randint(0, 9, (4, 6))
    labels[0, :5] = IGNORE  # uneven token counts are where per-micro-batch means go wrong
    full = ce_loss(logits, labels)
    n_tokens = (labels != IGNORE).sum()
    accum = sum(ce_loss(logits[i:i + 2], labels[i:i + 2], reduction="sum") for i in (0, 2)) / n_tokens
    assert accum.item() == pytest.approx(full.item(), abs=1e-6)


def test_all_ignored_gives_zero_not_nan():
    logits = torch.randn(1, 3, 5)
    labels = torch.full((1, 3), IGNORE)
    assert ce_loss(logits, labels).item() == 0.0


def test_fp16_input_is_upcast():
    logits = torch.randn(1, 2, 5).half()
    assert ce_loss(logits, torch.tensor([[1, 2]])).dtype == torch.float32


def _probs_as_logits(ps):
    return torch.tensor([[ps]]).log()


def test_fkl_of_identical_distributions_is_zero():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 11)
    labels = torch.zeros(2, 4, dtype=torch.long)
    assert fkl_loss(logits, logits.clone(), labels).item() == pytest.approx(0.0, abs=1e-6)


def test_fkl_hand_computed_value():
    # chapter 3: p = [0.7, 0.2, 0.1], q = [0.5, 0.3, 0.2] -> 0.0851
    teacher = _probs_as_logits([0.7, 0.2, 0.1])
    student = _probs_as_logits([0.5, 0.3, 0.2])
    loss = fkl_loss(student, teacher, torch.tensor([[0]]))
    assert loss.item() == pytest.approx(0.08512, abs=1e-5)


def test_fkl_is_ce_minus_teacher_entropy():
    # CE(p, q) = H(p) + KL(p || q), with a one-hot p this is plain ce
    student = _probs_as_logits([0.5, 0.3, 0.2])
    teacher = torch.tensor([[[30.0, 0.0, 0.0]]])
    labels = torch.tensor([[0]])
    assert fkl_loss(student, teacher, labels).item() == pytest.approx(ce_loss(student, labels).item(), abs=1e-5)


def test_fkl_gradient_is_q_minus_p():
    student = _probs_as_logits([0.5, 0.3, 0.2]).requires_grad_()
    teacher = _probs_as_logits([0.7, 0.2, 0.1])
    fkl_loss(student, teacher, torch.tensor([[0]])).backward()
    assert torch.allclose(student.grad[0, 0], torch.tensor([-0.2, 0.1, 0.1]), atol=1e-6)


def test_fkl_matches_torch_kl_div_with_mask_and_temperature():
    torch.manual_seed(0)
    s, t = torch.randn(2, 5, 9), torch.randn(2, 5, 9)
    labels = torch.randint(0, 9, (2, 5))
    labels[0, :3] = IGNORE
    T = 2.0
    per_token = F.kl_div(
        F.log_softmax(s / T, -1), F.log_softmax(t / T, -1), log_target=True, reduction="none"
    ).sum(-1)
    mask = labels != IGNORE
    ref = per_token[mask].mean() * T**2
    assert fkl_loss(s, t, labels, temperature=T).item() == pytest.approx(ref.item(), abs=1e-5)


def test_fkl_sends_no_gradient_to_teacher():
    s = torch.randn(1, 3, 5, requires_grad=True)
    t = torch.randn(1, 3, 5, requires_grad=True)
    fkl_loss(s, t, torch.zeros(1, 3, dtype=torch.long)).backward()
    assert s.grad is not None
    assert t.grad is None


def test_fkl_survives_zero_probability_teacher_tokens():
    # top-k caching will hand us exactly this: tokens the teacher gives no mass
    teacher = torch.tensor([[[2.0, 1.0, float("-inf")]]])
    student = torch.randn(1, 1, 3, requires_grad=True)
    loss = fkl_loss(student, teacher, torch.tensor([[0]]))
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
