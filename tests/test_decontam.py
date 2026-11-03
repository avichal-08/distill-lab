from distill_lab.data.decontam import build_index, find_overlaps, ngrams, too_short, words

TEST_Q = "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins with four."


def rec(id_, question, solution=None):
    return {"id": id_, "question": question, "solution": solution}


def test_words_ignore_case_and_punctuation():
    assert words("Janet's ducks, LAY 16-eggs!") == ["janet", "s", "ducks", "lay", "16", "eggs"]


def test_ngram_count():
    assert len(ngrams("a b c d e", n=3)) == 3
    assert ngrams("a b", n=3) == set()


def test_exact_copy_is_found():
    index = build_index([rec("t1", TEST_Q)], ("question",))
    hits = find_overlaps([rec("x1", TEST_Q)], ("question",), index)
    assert hits[0]["id"] == "x1" and hits[0]["test_ids"] == ["t1"]


def test_copy_hidden_in_a_solution_with_different_formatting_is_found():
    index = build_index([rec("t1", TEST_Q)], ("question",))
    sneaky = rec("x1", "Some other question?", "As the problem says: JANET'S ducks lay 16 eggs per day; she eats three for breakfast every morning...")
    assert find_overlaps([sneaky], ("question", "solution"), index)


def test_unrelated_text_is_clean():
    index = build_index([rec("t1", TEST_Q)], ("question",))
    other = rec("x1", "Tom has 5 apples and buys 3 more at the store near his house on Monday. How many now?")
    assert find_overlaps([other], ("question",), index) == []


def test_twelve_shared_words_is_not_enough():
    index = build_index([rec("t1", TEST_Q)], ("question",))
    twelve = " ".join(words(TEST_Q)[:12])
    assert find_overlaps([rec("x1", twelve + " and then something else entirely different")], ("question",), index) == []


def test_short_questions_are_reported():
    recs = [rec("short", "What is 2 plus 2?"), rec("long", TEST_Q)]
    assert too_short(recs, ("question",)) == ["short"]
