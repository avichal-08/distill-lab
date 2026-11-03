import pytest
import torch

from distill_lab.analysis.topk_coverage import topk_stats


def test_uniform_teacher_mass_is_k_over_v():
    logits = torch.zeros(1, 3, 100)
    mask = torch.ones(1, 3, dtype=torch.bool)
    stats = topk_stats(logits, mask, [10, 50])
    assert stats[10]["mass_mean"] == pytest.approx(0.1, abs=1e-5)
    assert stats[50]["mass_mean"] == pytest.approx(0.5, abs=1e-5)
    assert stats[50]["frac_tail_over_1pct"] == 1.0


def test_whole_vocab_keeps_everything():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 20)
    stats = topk_stats(logits, torch.ones(2, 4, dtype=torch.bool), [20])
    assert stats[20]["mass_mean"] == pytest.approx(1.0, abs=1e-5)
    assert stats[20]["frac_tail_over_1pct"] == 0.0


def test_masked_positions_are_ignored():
    logits = torch.zeros(1, 2, 10)
    logits[0, 1, 0] = 50.0  # position 1 is fully peaked, position 0 is uniform
    mask = torch.tensor([[False, True]])
    assert topk_stats(logits, mask, [1])[1]["mass_mean"] == pytest.approx(1.0, abs=1e-5)


def test_error_shrinks_as_k_grows_and_vanishes_at_v():
    torch.manual_seed(0)
    t = torch.randn(2, 8, 200) * 3
    s = t + torch.randn_like(t)
    mask = torch.ones(2, 8, dtype=torch.bool)
    stats = topk_stats(t, mask, [4, 32, 200], s)
    for name in ("fkl_loss", "rkl_loss", "jsd_loss"):
        errs = [abs(stats[k][f"{name}_rel_err"]) for k in (4, 32, 200)]
        assert errs[0] > errs[1] > errs[2]
        assert errs[2] < 1e-3
