import pytest

from distill_lab.teacher.generate import select, trim_solution

Q1 = {"id": "gsm8k-train-7", "question": "What is 48 + 24?", "answer": "72"}
Q2 = {"id": "gsm8k-train-8", "question": "What is 3 * 5?", "answer": "15"}


def test_trim_keeps_everything_up_to_the_hash_line():
    assert trim_solution(" 48 + 24 = 72\n#### 72\n\nNow a harder one: 5 + 5 = 10") == "48 + 24 = 72\n#### 72"


def test_trim_cuts_at_the_next_question_first():
    assert trim_solution(" 48 + 24 = 72\nQuestion: what is 2+2?\n#### 4") is None


def test_trim_rejects_solutions_without_a_final_line():
    assert trim_solution(" 48 + 24 = 72, so the answer is 72.") is None


def test_select_keeps_correct_distinct_solutions():
    samples = [
        [" 48 + 24 = 72\n#### 72", " 48 + 24 = 72\n#### 72"],  # same text twice
        [" 3 * 5 = 16\n#### 16", " 3 * 5 = 15\n#### 15"],
    ]
    kept, stats = select([Q1, Q2], samples)
    assert [k["id"] for k in kept] == ["gsm8k-train-7-s0", "gsm8k-train-8-s1"]
    assert kept[1]["solution"] == "3 * 5 = 15\n#### 15"
    assert kept[0]["question_id"] == "gsm8k-train-7"
    assert stats["sample_accuracy"] == pytest.approx(3 / 4)
    assert stats["duplicates_dropped"] == 1
    assert stats["questions_never_solved"] == 0


def test_select_counts_questions_the_teacher_never_solves():
    kept, stats = select([Q1, Q2], [[" 70\n#### 70"], [" 15\n#### 15"]])
    assert len(kept) == 1
    assert stats["questions_never_solved"] == 1
    assert stats["questions_with_a_correct_solution"] == 1


def test_select_ignores_an_answer_hidden_after_rambling():
    # right number, but only after the model stopped giving a final answer
    kept, _ = select([Q1], [[" maybe 70? or 72, not sure"]])
    assert kept == []
