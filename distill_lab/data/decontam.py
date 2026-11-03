"""check that no test question shows up inside the training data.

any 13-word run shared between a training text and a test text counts as a hit.

    python -m distill_lab.data.decontam --data cache/data
    python -m distill_lab.data.decontam --data cache/data --drop
    python -m distill_lab.data.decontam --data cache/data --extra cache/teacher/gsm8k_train.jsonl
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from distill_lab.data.prepare import load_split, save

N = 13
_WORD = re.compile(r"[a-z0-9]+")


def words(text: str) -> list[str]:
    # punctuation, case and spacing differences shouldn't hide a copy
    return _WORD.findall(text.lower())


def ngrams(text: str, n: int = N) -> set[tuple[str, ...]]:
    w = words(text)
    return {tuple(w[i : i + n]) for i in range(len(w) - n + 1)}


def _texts(rec: dict, fields: tuple[str, ...]) -> str:
    return " ".join(rec[f] for f in fields if rec.get(f))


def build_index(test_recs: list[dict], fields: tuple[str, ...], n: int = N) -> dict[tuple, set[str]]:
    index: dict[tuple, set[str]] = defaultdict(set)
    for r in test_recs:
        for g in ngrams(_texts(r, fields), n):
            index[g].add(r["id"])
    return index


def find_overlaps(train_recs: list[dict], fields: tuple[str, ...], index: dict, n: int = N) -> list[dict]:
    hits = []
    for r in train_recs:
        shared: dict[str, int] = defaultdict(int)
        for g in ngrams(_texts(r, fields), n):
            for test_id in index.get(g, ()):
                shared[test_id] += 1
        if shared:
            hits.append({"id": r["id"], "test_ids": sorted(shared), "shared_ngrams": max(shared.values())})
    return hits


def too_short(recs: list[dict], fields: tuple[str, ...], n: int = N) -> list[str]:
    # fewer than n words means no n-grams, so this check can't see them at all
    return [r["id"] for r in recs if len(words(_texts(r, fields))) < n]


def drop_contaminated(train: list[dict], tests: list[list[dict]], n: int = N) -> tuple[list[dict], list[str]]:
    flagged: set[str] = set()
    for test_recs in tests:
        index = build_index(test_recs, ("question",), n)
        flagged |= {h["id"] for h in find_overlaps(train, ("question", "solution"), index, n)}
    return [r for r in train if r["id"] not in flagged], sorted(flagged)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="cache/data")
    ap.add_argument("--extra", default=None, help="another jsonl of training texts, e.g. teacher solutions")
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--drop", action="store_true", help="rewrite gsm8k_train.jsonl without the flagged records")
    args = ap.parse_args()

    tests = {name: load_split(args.data, name) for name in ("gsm8k_test", "svamp_test")}
    train = {"gsm8k_train": load_split(args.data, "gsm8k_train")}
    if args.extra:
        with open(args.extra, encoding="utf-8") as f:
            train[Path(args.extra).stem] = [json.loads(line) for line in f]

    report = {"n": args.n, "too_short": {k: too_short(v, ("question",), args.n) for k, v in tests.items()}, "hits": {}}
    for test_name, test_recs in tests.items():
        index = build_index(test_recs, ("question",), args.n)
        for train_name, train_recs in train.items():
            hits = find_overlaps(train_recs, ("question", "solution"), index, args.n)
            report["hits"][f"{train_name} vs {test_name}"] = hits
            print(f"{train_name:14s} vs {test_name:11s}: {len(hits):4d} training texts share a {args.n}-gram")

    for k, ids in report["too_short"].items():
        print(f"{k}: {len(ids)} questions shorter than {args.n} words, unchecked")

    if args.drop:
        kept, report["dropped"] = drop_contaminated(train["gsm8k_train"], list(tests.values()), args.n)
        save({"gsm8k_train": kept}, Path(args.data))
        print(f"dropped {len(report['dropped'])}, gsm8k_train now has {len(kept)} records")

    out = Path(args.data) / "decontam_report.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"report: {out}")


if __name__ == "__main__":
    main()
