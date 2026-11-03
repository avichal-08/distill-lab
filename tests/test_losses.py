import math

import pytest
import torch
import torch.nn.functional as F

from distill_lab.train.losses import IGNORE, TopK, ce_loss, fkl_loss, jsd_loss, rkl_loss, shift


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


ONE = torch.tensor([[0]])


def test_rkl_hand_computed_value():
    # chapter 3: KL(q || p) with the same p, q -> 0.0920
    loss = rkl_loss(_probs_as_logits([0.5, 0.3, 0.2]), _probs_as_logits([0.7, 0.2, 0.1]), ONE)
    assert loss.item() == pytest.approx(0.09203, abs=1e-5)


def test_rkl_matches_torch_kl_div_with_args_swapped():
    torch.manual_seed(0)
    s, t = torch.randn(2, 5, 9), torch.randn(2, 5, 9)
    labels = torch.randint(0, 9, (2, 5))
    labels[1, 2:] = IGNORE
    per_token = F.kl_div(F.log_softmax(t, -1), F.log_softmax(s, -1), log_target=True, reduction="none").sum(-1)
    ref = per_token[labels != IGNORE].mean()
    assert rkl_loss(s, t, labels).item() == pytest.approx(ref.item(), abs=1e-5)


def test_forward_covers_modes_reverse_picks_one():
    # chapter 3's bimodal teacher: forward kl prefers the spread student, reverse kl the one-mode student
    teacher = _probs_as_logits([0.49, 0.02, 0.49])
    spread = _probs_as_logits([0.25, 0.50, 0.25])
    one_mode = _probs_as_logits([0.96, 0.02, 0.02])
    f_spread, f_mode = fkl_loss(spread, teacher, ONE).item(), fkl_loss(one_mode, teacher, ONE).item()
    r_spread, r_mode = rkl_loss(spread, teacher, ONE).item(), rkl_loss(one_mode, teacher, ONE).item()
    assert (f_spread, f_mode) == pytest.approx((0.595, 1.238), abs=1e-3)
    assert (r_spread, r_mode) == pytest.approx((1.273, 0.581), abs=1e-3)
    assert f_spread < f_mode and r_mode < r_spread


def test_jsd_hand_computed_value():
    # chapter 3: 0.0219
    loss = jsd_loss(_probs_as_logits([0.5, 0.3, 0.2]), _probs_as_logits([0.7, 0.2, 0.1]), ONE)
    assert loss.item() == pytest.approx(0.02190, abs=1e-5)


def test_jsd_is_symmetric_at_half():
    torch.manual_seed(0)
    a, b = torch.randn(1, 4, 6), torch.randn(1, 4, 6)
    labels = torch.zeros(1, 4, dtype=torch.long)
    assert jsd_loss(a, b, labels).item() == pytest.approx(jsd_loss(b, a, labels).item(), abs=1e-6)


def test_jsd_is_bounded_by_ln2():
    # no overlap at all is the worst case
    teacher = torch.tensor([[[30.0, -30.0]]])
    student = torch.tensor([[[-30.0, 30.0]]])
    assert jsd_loss(student, teacher, ONE).item() == pytest.approx(math.log(2), abs=1e-4)


@pytest.mark.parametrize("loss_fn", [rkl_loss, jsd_loss])
def test_identical_is_zero_and_teacher_gets_no_gradient(loss_fn):
    torch.manual_seed(0)
    s = torch.randn(2, 3, 7, requires_grad=True)
    t = s.detach().clone().requires_grad_()
    labels = torch.zeros(2, 3, dtype=torch.long)
    loss = loss_fn(s, t, labels)
    loss.backward()
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert t.grad is None


def test_jsd_survives_zero_probability_teacher_tokens():
    teacher = torch.tensor([[[2.0, 1.0, float("-inf")]]])
    student = torch.randn(1, 1, 3, requires_grad=True)
    loss = jsd_loss(student, teacher, ONE)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(student.grad).all()


def _topk_of(teacher_logits, k):
    logprobs, ids = F.log_softmax(teacher_logits, -1).topk(k, dim=-1)
    return TopK(ids=ids, logprobs=logprobs)


def test_topk_hand_computed_tail_bucket():
    # chapter 4: teacher top-2 = [0.60, 0.25], tail 0.15. student gives those ids
    # [0.40, 0.30] and spreads its 0.30 tail over the rest -> 0.0937
    teacher = TopK(ids=torch.tensor([[[0, 1]]]), logprobs=torch.tensor([[[0.60, 0.25]]]).log())
    student = _probs_as_logits([0.40, 0.30, 0.10, 0.10, 0.10])
    assert fkl_loss(student, teacher, ONE).item() == pytest.approx(0.09373, abs=1e-5)


def test_topk_tail_shape_is_invisible():
    # only the tail's total counts, not how the student spreads it
    teacher = TopK(ids=torch.tensor([[[0, 1]]]), logprobs=torch.tensor([[[0.60, 0.25]]]).log())
    a = _probs_as_logits([0.40, 0.30, 0.10, 0.10, 0.10])
    b = _probs_as_logits([0.40, 0.30, 0.28, 0.01, 0.01])
    assert fkl_loss(a, teacher, ONE).item() == pytest.approx(fkl_loss(b, teacher, ONE).item(), abs=1e-6)


@pytest.mark.parametrize("loss_fn", [fkl_loss, rkl_loss, jsd_loss])
def test_topk_over_whole_vocab_matches_dense(loss_fn):
    torch.manual_seed(0)
    s, t = torch.randn(2, 4, 12), torch.randn(2, 4, 12)
    labels = torch.zeros(2, 4, dtype=torch.long)
    dense = loss_fn(s, t, labels).item()
    # k = V leaves an empty tail, which only the floor fills
    assert loss_fn(s, _topk_of(t, 12), labels).item() == pytest.approx(dense, abs=1e-4)


@pytest.mark.parametrize("loss_fn", [fkl_loss, rkl_loss, jsd_loss])
def test_topk_32_is_close_to_dense_on_a_peaked_teacher(loss_fn):
    # lm distributions are peaked, so 32 of 1000 ids hold nearly all the mass
    torch.manual_seed(0)
    t = torch.randn(4, 16, 1000) * 4
    s = t + torch.randn_like(t)
    labels = torch.zeros(4, 16, dtype=torch.long)
    dense = loss_fn(s, t, labels).item()
    approx = loss_fn(s, _topk_of(t, 32), labels).item()
    assert abs(approx - dense) / dense < 0.1


@pytest.mark.parametrize("loss_fn", [fkl_loss, rkl_loss, jsd_loss])
def test_topk_gradients_are_finite_and_skip_the_teacher(loss_fn):
    torch.manual_seed(0)
    s = torch.randn(2, 3, 50, requires_grad=True)
    teacher = _topk_of(torch.randn(2, 3, 50) * 5, 8)
    teacher.logprobs.requires_grad_()
    loss = loss_fn(s, teacher, torch.zeros(2, 3, dtype=torch.long), temperature=2.0)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(s.grad).all()
    assert teacher.logprobs.grad is None


def test_topk_handles_fp16_cache_summing_past_one():
    # rounding in the cache can make the top-k mass exceed 1, tail would go negative
    logprobs = torch.tensor([[[0.7, 0.31]]]).log().half()
    teacher = TopK(ids=torch.tensor([[[0, 1]]]), logprobs=logprobs)
    loss = rkl_loss(torch.randn(1, 1, 6), teacher, ONE)
    assert torch.isfinite(loss) and loss.item() >= 0
