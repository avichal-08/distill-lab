"""load gsm8k and svamp into one record format and save them.

    python -m distill_lab.data.prepare --out cache/data
    python -m distill_lab.data.prepare --out cache/data --push <hf-user>/distill-lab-data
"""

import argparse
import json
import re
from pathlib import Path

PROMPT = "Question: {question}\nAnswer:"
# the first few gsm8k train problems, used as worked examples for base models
SHOT_IDS = [0, 1, 2, 3]

_CALC = re.compile(r"<<[^>]*>>")
_FINAL = re.compile(r"####\s*(.+)\s*$")


def canonical_number(text: str) -> str:
    """'$1,200.00' -> '1200', '27.0' -> '27', '0.50' -> '0.5'. raises ValueError if not a number"""
    t = text.strip().replace(",", "").replace("$", "").rstrip(".")
    value = float(t)
    if value == int(value):
        return str(int(value))
    return repr(value)


def clean_solution(raw: str) -> str:
    # gsm8k solutions carry calculator annotations like <<48/2=24>>, models shouldn't learn to write those
    return _CALC.sub("", raw).strip()


def final_answer(solution: str) -> str:
    m = _FINAL.search(solution)
    if m is None:
        raise ValueError(f"no '#### answer' line in: {solution[-80:]!r}")
    return canonical_number(m.group(1))


def gsm8k_record(row: dict, split: str, i: int) -> dict:
    solution = clean_solution(row["answer"])
    return {
        "id": f"gsm8k-{split}-{i}",
        "source": "gsm8k",
        "question": row["question"].strip(),
        "solution": solution,
        "answer": final_answer(solution),
    }


def svamp_record(row: dict) -> dict:
    body = row["Body"].strip()
    # some bodies end without punctuation and run straight into the question
    if body and body[-1] not in ".?!,":
        body += "."
    return {
        "id": f"svamp-{row['ID']}",
        "source": "svamp",
        "question": f"{body} {row['Question'].strip()}",
        "solution": None,
        "answer": canonical_number(str(row["Answer"])),
    }


def format_prompt(question: str) -> str:
    return PROMPT.format(question=question)


def format_example(rec: dict) -> tuple[str, str]:
    """(prompt, target). the leading space keeps the tokenizer split the same as at generation time"""
    return format_prompt(rec["question"]), " " + rec["solution"]


def few_shot_prompt(question: str, shots: list[dict]) -> str:
    worked = ["".join(format_example(s)) for s in shots]
    return "\n\n".join(worked + [format_prompt(question)])


def load_all() -> dict[str, list[dict]]:
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main")
    svamp = load_dataset("ChilleD/SVAMP")
    return {
        "gsm8k_train": [gsm8k_record(r, "train", i) for i, r in enumerate(gsm["train"])],
        "gsm8k_test": [gsm8k_record(r, "test", i) for i, r in enumerate(gsm["test"])],
        # svamp's own train/test split doesn't matter here, all 1000 are eval only
        "svamp_test": [svamp_record(r) for part in ("train", "test") for r in svamp[part]],
    }


def shots(gsm8k_train: list[dict]) -> list[dict]:
    return [gsm8k_train[i] for i in SHOT_IDS]


def save(splits: dict[str, list[dict]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in splits.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_split(out: Path, name: str) -> list[dict]:
    with open(Path(out) / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def push(splits: dict[str, list[dict]], repo_id: str) -> None:
    from datasets import Dataset, DatasetDict

    DatasetDict({k: Dataset.from_list(v) for k, v in splits.items()}).push_to_hub(repo_id, private=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cache/data")
    ap.add_argument("--push", default=None, help="hf dataset repo, e.g. you/distill-lab-data")
    args = ap.parse_args()

    from distill_lab.data.decontam import drop_contaminated

    splits = load_all()
    splits["gsm8k_train"], dropped = drop_contaminated(splits["gsm8k_train"], [splits["gsm8k_test"], splits["svamp_test"]])
    print(f"dropped {len(dropped)} training records that overlap a test question: {', '.join(dropped)}")
    save(splits, Path(args.out))
    for name, recs in splits.items():
        print(f"{name:12s} {len(recs):5d} records")
    print("\nexample 4-shot prompt tail:\n" + few_shot_prompt(splits["gsm8k_test"][0]["question"], shots(splits["gsm8k_train"]))[-400:])
    if args.push:
        push(splits, args.push)
        print(f"\npushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
