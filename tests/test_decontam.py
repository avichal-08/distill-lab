from distill_lab.data.decontam import build_index, drop_contaminated, find_overlaps, ngrams, too_short, words

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


def test_drop_contaminated_checks_every_test_set():
    train = [rec("clean", "Tom has 5 apples."), rec("copy_a", TEST_Q), rec("copy_b", "x", "Mia baked 40 cookies on Sunday and gave half of them away to her three neighbours.")]
    test_a = [rec("ta", TEST_Q)]
    test_b = [rec("tb", "Mia baked 40 cookies on Sunday and gave half of them away to her three neighbours. How many are left?")]
    kept, dropped = drop_contaminated(train, [test_a, test_b])
    assert [r["id"] for r in kept] == ["clean"]
    assert dropped == ["copy_a", "copy_b"]
