import pytest

from distill_lab.eval.extract import extract_answer, is_correct


@pytest.mark.parametrize("text, want", [
    # the format we train on
    ("48/2 = 24 in May, 48 + 24 = 72 total.\n#### 72", "72"),
    ("####72", "72"),
    ("#### 72.", "72"),
    ("#### $1,250.00", "1250"),
    ("#### 18 dollars", "18"),
    ("#### -3", "-3"),
    ("#### 0.5", "0.5"),
    ("#### 72 (48 + 24)", "72"),
    # base models keep going and write the next problem, with its own answer
    ("so 72 clips.\n#### 72\n\nQuestion: Tom has 5 apples...\nAnswer: 5 + 3 = 8\n#### 8", "72"),
    ("The total is 72.\n\nQuestion: A train travels 300 miles", "72"),
    # other ways models state the answer
    ("Total = 45, so the answer is 45.", "45"),
    ("The answer is: $1,200", "1200"),
    ("Answer: 16", "16"),
    ("The final answer is **18**.", "18"),
    ("Thus the result is \\boxed{42}.", "42"),
    ("<think>maybe 10? no, 12</think>\nShe has 12 left.\n#### 12", "12"),
    # nothing explicit, take the last number
    ("48 + 24 = 72 clips altogether", "72"),
    ("It went up by 50%", "50"),
    ("30-5 = 25", "25"),
    ("she ends with **7** apples", "7"),
    # nothing to find
    ("I don't know.", None),
    ("", None),
])
def test_extract_answer(text, want):
    assert extract_answer(text) == want


def test_minus_inside_arithmetic_is_not_a_negative_number():
    assert extract_answer("the change is 30-5") == "5"


def test_hash_line_beats_numbers_after_it():
    assert extract_answer("#### 9\nwhich took 3 steps") == "9"


@pytest.mark.parametrize("pred, gold, ok", [
    ("72", "72", True),
    ("72", "73", False),
    ("0.5", "0.50", True),
    (None, "72", False),
])
def test_is_correct(pred, gold, ok):
    assert is_correct(pred, gold) is ok
