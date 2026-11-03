import json

import pytest

from distill_lab.eval.run_eval import build_prompts, evaluate, score, truncate_at_stop

RECS = [
    {"id": "q1", "question": "What is 2+2?", "answer": "4"},
    {"id": "q2", "question": "What is 3+3?", "answer": "6"},
    {"id": "q3", "question": "What is 5+5?", "answer": "10"},
    {"id": "q4", "question": "What is 1+1?", "answer": "2"},
]
SHOT = {"id": "s", "question": "What is 1+2?", "solution": "1 + 2 = 3\n#### 3", "answer": "3"}


def test_zero_shot_prompt():
    assert build_prompts(RECS[:1], []) == ["Question: What is 2+2?\nAnswer:"]


def test_few_shot_prompt_has_the_examples_first():
    p = build_prompts(RECS[:1], [SHOT, SHOT])[0]
    assert p.count("#### 3") == 2
    assert p.endswith("Question: What is 2+2?\nAnswer:")


def test_truncate_at_stop():
    assert truncate_at_stop(" 4\n#### 4\n\nQuestion: next one") == " 4\n#### 4\n\n"
    assert truncate_at_stop("no stop here") == "no stop here"


def test_score_counts_and_interval():
    outs = ["#### 4", "#### 7", "the answer is 10", "no idea"]
    rows, s = score(RECS, outs)
    assert [r["correct"] for r in rows] == [True, False, True, False]
    assert s["accuracy"] == 0.5 and s["n"] == 4 and s["no_answer"] == 1
    # 1.96 * sqrt(0.25 / 4)
    assert s["ci95"] == pytest.approx(0.49)


def test_score_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        score(RECS, ["#### 4"])


def test_evaluate_writes_rows_and_summary(tmp_path):
    seen = []

    def fake_generate(prompts):
        seen.extend(prompts)
        return ["#### 4", "#### 6", "#### 0", "#### 2"][: len(prompts)]

    summaries = evaluate({"toy_test": RECS}, [SHOT], fake_generate, tmp_path / "run")
    assert summaries["toy_test"]["accuracy"] == 0.75
    assert len(seen) == 4 and all(p.startswith("Question: What is 1+2?") for p in seen)

    rows = [json.loads(l) for l in (tmp_path / "run" / "toy_test.jsonl").read_text().splitlines()]
    assert [r["pred"] for r in rows] == ["4", "6", "0", "2"]
    assert json.loads((tmp_path / "run" / "summary.json").read_text())["toy_test"]["n"] == 4
