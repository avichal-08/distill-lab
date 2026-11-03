import pytest

from distill_lab.data.prepare import (
    canonical_number,
    clean_solution,
    few_shot_prompt,
    final_answer,
    format_example,
    gsm8k_record,
    svamp_record,
)

GSM_ROW = {
    "question": "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? ",
    "answer": "Natalia sold 48/2 = <<48/2=24>>24 clips in May.\nNatalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May.\n#### 72",
}


@pytest.mark.parametrize("raw, want", [
    ("72", "72"), ("1,200", "1200"), ("$5", "5"), ("27.0", "27"),
    ("0.50", "0.5"), ("-3", "-3"), (" 18 ", "18"), ("18.", "18"),
])
def test_canonical_number(raw, want):
    assert canonical_number(raw) == want


def test_canonical_number_rejects_words():
    with pytest.raises(ValueError):
        canonical_number("seventy two")


def test_calculator_annotations_are_stripped():
    out = clean_solution(GSM_ROW["answer"])
    assert "<<" not in out
    assert out.startswith("Natalia sold 48/2 = 24 clips in May.")


def test_final_answer_reads_the_hash_line():
    assert final_answer("so it costs $1,250.\n#### 1,250") == "1250"


def test_final_answer_needs_the_hash_line():
    with pytest.raises(ValueError):
        final_answer("the answer is 72")


def test_gsm8k_record():
    rec = gsm8k_record(GSM_ROW, "train", 0)
    assert rec["id"] == "gsm8k-train-0"
    assert rec["answer"] == "72"
    assert rec["question"].endswith("April and May?")  # trailing space stripped
    assert rec["solution"].endswith("#### 72")


def test_svamp_record_joins_body_and_question():
    row = {"ID": "chal-1", "Body": "There are 62 birds", "Question": "How many are left?", "Answer": 27.0}
    rec = svamp_record(row)
    assert rec["question"] == "There are 62 birds. How many are left?"
    assert rec["answer"] == "27"
    assert rec["solution"] is None


def test_format_example_splits_prompt_and_target():
    prompt, target = format_example(gsm8k_record(GSM_ROW, "train", 0))
    assert prompt.startswith("Question: Natalia")
    assert prompt.endswith("\nAnswer:")
    assert target.startswith(" Natalia sold")


def test_few_shot_prompt_ends_with_an_open_answer():
    shot = gsm8k_record(GSM_ROW, "train", 0)
    p = few_shot_prompt("What is 2+2?", [shot, shot])
    assert p.count("#### 72") == 2
    assert p.endswith("Question: What is 2+2?\nAnswer:")
    # worked examples are separated by a blank line so the model sees where one ends
    assert "#### 72\n\nQuestion:" in p
