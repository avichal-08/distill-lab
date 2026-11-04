import copy

import pytest
import torch

from distill_lab.config import load_config
from distill_lab.data.tokens import tokenize_record
from distill_lab.train.data import build_examples, collate, label_tokens, load_records
from distill_lab.train.losses import IGNORE
from distill_lab.train.trainer import train, warmup_cosine
from tests.test_teacher_logits import CharTok, tiny_model

RECS = [{"id": f"q{i}", "question_id": f"q{i}", "question": f"add {i} and {i}", "solution": f"{i}+{i}={2 * i}\n#### {2 * i}"}
        for i in range(8)]


def tiny_cfg(kind: str, **train_over) -> dict:
    cfg = load_config("configs/base.yaml")
    cfg["loss"]["kind"] = kind
    cfg["train"].update(micro_batch=4, grad_accum=2, epochs=60, lr=3e-3, warmup_ratio=0.05,
                        grad_ckpt=False, fp16=False, log_every=1000, **train_over)
    return cfg


def teacher_topk(tok, recs, k=8):
    """a confident made-up teacher: 0.9 on the real next token, the rest spread over k-1 others.
    reachable by the student, so a working loop has to get its kd loss near zero"""
    out = {}
    for r in recs:
        ex = tokenize_record(tok, r, 128)
        nxt = torch.tensor([l for l in ex["labels"][1:] if l != IGNORE])
        others = (nxt[:, None] + torch.arange(1, k)[None, :]) % 64
        ids = torch.cat([nxt[:, None], others], dim=1).int()
        lp = torch.full((len(nxt), k), torch.tensor(0.1 / (k - 1)).log().item())
        lp[:, 0] = torch.tensor(0.9).log()
        out[r["id"]] = (lp.half(), ids)
    return out


def test_collate_lines_teacher_rows_up_with_graded_positions():
    tok = CharTok()
    topk = {r["id"]: (torch.full((n, 3), -1.0).half(), torch.arange(n * 3).view(n, 3).int())
            for r in RECS[:2] for n in [sum(l != IGNORE for l in tokenize_record(tok, r, 128)["labels"][1:])]}
    b = collate(build_examples(tok, RECS[:2], 128, topk), pad_id=0)
    graded = b["labels"][:, 1:] != IGNORE
    assert b["teacher"].ids.shape[:2] == graded.shape
    for i, r in enumerate(RECS[:2]):
        assert torch.equal(b["teacher"].ids[i][graded[i]], topk[r["id"]][1].long())
        assert (b["teacher"].ids[i][~graded[i]] == 0).all()


def test_build_examples_refuses_misaligned_teacher_rows():
    tok = CharTok()
    bad = {"q0": (torch.zeros(3, 4).half(), torch.zeros(3, 4).int())}
    with pytest.raises(ValueError, match="teacher rows"):
        build_examples(tok, RECS[:1], 128, bad)


def test_warmup_then_cosine_to_zero():
    f = warmup_cosine(100, 0.1)
    assert f(0) == pytest.approx(0.1) and f(9) == pytest.approx(1.0)
    assert f(55) == pytest.approx(0.5, abs=0.02) and f(100) == pytest.approx(0.0, abs=1e-6)


def test_n_questions_cuts_by_question(tmp_path):
    (tmp_path / "teacher").mkdir()
    lines = [f'{{"id": "q{i // 2}-s{i % 2}", "question_id": "q{i // 2}", "question": "x", "solution": "#### 1", "answer": "1"}}'
             for i in range(10)]
    (tmp_path / "teacher" / "gsm8k_train.jsonl").write_text("\n".join(lines))
    recs = load_records({"source": "teacher", "n_questions": 2}, tmp_path / "data", seed=0)
    assert len({r["question_id"] for r in recs}) == 2 and len(recs) == 4


@pytest.mark.parametrize("kind", ["ce", "fkl", "rkl", "jsd"])
def test_tiny_student_can_overfit(kind):
    # E3 on cpu: if the loop can't memorise 8 examples, something in it is broken
    tok = CharTok()
    teacher = tiny_model()
    student = copy.deepcopy(teacher)
    torch.manual_seed(1)
    for p in student.parameters():
        p.data += 0.5 * torch.randn_like(p)
    topk = None if kind == "ce" else teacher_topk(tok, RECS)
    examples = build_examples(tok, RECS, 128, topk)
    hist = train(tiny_cfg(kind), student, tok, examples, log=lambda row: None)
    first, last = hist[0], hist[-1]
    key = "ce" if kind == "ce" else "kd"
    # jsd is capped at ln 2, its gradient is flat while the student is far off, so it moves slower
    assert last[key] < (0.35 if kind == "jsd" else 0.25) * first[key], (first, last)
    assert all(torch.isfinite(torch.tensor(h["grad_norm"])) for h in hist)


def test_label_tokens_counts_shifted_labels():
    ex = build_examples(CharTok(), RECS[:1], 128)
    assert label_tokens(ex) == sum(l != IGNORE for l in ex[0]["labels"][1:])
