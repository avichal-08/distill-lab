"""the teacher solves every training question a few times, we keep the solutions with the right answer.

    python -m distill_lab.teacher.generate --out cache/teacher
    python -m distill_lab.teacher.generate --out cache/teacher --push <hf-user>/distill-lab-teacher
    python -m distill_lab.teacher.generate --limit 20 --backend hf    # laptop smoke test
"""

import argparse
import json
import re
import statistics
from pathlib import Path

from distill_lab.config import load_config
from distill_lab.data.decontam import drop_contaminated
from distill_lab.data.prepare import few_shot_prompt, load_split, save, shots
from distill_lab.eval.extract import extract_answer, is_correct
from distill_lab.eval.run_eval import STOP, truncate_at_stop

_HASH_LINE = re.compile(r"^(.*?####[^\n]*)", re.DOTALL)


def trim_solution(text: str) -> str | None:
    """the solution up to and including its '#### n' line, or None if it never gives one.
    anything after that line is the model rambling on"""
    m = _HASH_LINE.match(truncate_at_stop(text))
    return m.group(1).strip() if m else None


def select(records: list[dict], samples: list[list[str]]) -> tuple[list[dict], dict]:
    """keep each distinct correct solution. records[i] was answered by samples[i]"""
    kept, n_correct_per_q = [], []
    for rec, outs in zip(records, samples, strict=True):
        seen, n_ok = set(), 0
        for j, out in enumerate(outs):
            sol = trim_solution(out)
            if sol is None or not is_correct(extract_answer(sol), rec["answer"]):
                continue
            n_ok += 1
            if sol in seen:
                continue
            seen.add(sol)
            kept.append({"id": f"{rec['id']}-s{j}", "question_id": rec["id"], "question": rec["question"],
                         "solution": sol, "answer": rec["answer"]})
        n_correct_per_q.append(n_ok)
    n_q, n_s = len(records), sum(len(o) for o in samples)
    stats = {
        "questions": n_q,
        "samples": n_s,
        "sample_accuracy": sum(n_correct_per_q) / n_s if n_s else 0.0,
        "questions_with_a_correct_solution": sum(c > 0 for c in n_correct_per_q),
        # the hard ones distillation will never see
        "questions_never_solved": sum(c == 0 for c in n_correct_per_q),
        "kept_solutions": len(kept),
        "duplicates_dropped": sum(n_correct_per_q) - len(kept),
        "median_solution_chars": statistics.median(len(k["solution"]) for k in kept) if kept else 0,
    }
    return kept, stats


def vllm_sampler(model: str, gen: dict, seed: int):
    from vllm import LLM, SamplingParams

    llm = LLM(model=model, dtype="half", seed=seed, max_model_len=4096, gpu_memory_utilization=0.9)
    params = SamplingParams(n=gen["n_samples"], temperature=gen["temperature"], top_p=gen["top_p"],
                            max_tokens=gen["max_new_tokens"], stop=STOP, seed=seed)

    def sample(prompts: list[str]) -> list[list[str]]:
        return [[c.text for c in o.outputs] for o in llm.generate(prompts, params)]

    return sample


def hf_sampler(model: str, gen: dict, seed: int):
    from distill_lab.eval.run_eval import hf_generator

    one = hf_generator(model, gen["max_new_tokens"], gen["temperature"], seed)

    def sample(prompts: list[str]) -> list[list[str]]:
        return [one([p] * gen["n_samples"]) for p in prompts]

    return sample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--data", default="cache/data")
    ap.add_argument("--out", default="cache/teacher")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--push", default=None, help="hf dataset repo for the kept solutions")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    gen = cfg["teacher_gen"]
    train = load_split(args.data, "gsm8k_train")
    examples = shots(train)[: gen["n_shot"]]
    shot_ids = {e["id"] for e in examples}
    # the worked examples sit in every prompt, solving them again teaches nothing
    records = [r for r in train if r["id"] not in shot_ids][: args.limit]

    make = vllm_sampler if args.backend == "vllm" else hf_sampler
    samples = make(cfg["teacher"], gen, cfg["seed"])([few_shot_prompt(r["question"], examples) for r in records])

    kept, stats = select(records, samples)
    tests = [load_split(args.data, "gsm8k_test"), load_split(args.data, "svamp_test")]
    kept, leaked = drop_contaminated(kept, tests)
    stats["dropped_for_test_overlap"] = leaked

    out = Path(args.out)
    save({"gsm8k_train": kept}, out)
    with open(out / "raw_samples.jsonl", "w", encoding="utf-8") as f:
        for r, outs in zip(records, samples):
            f.write(json.dumps({"id": r["id"], "answer": r["answer"], "samples": outs}, ensure_ascii=False) + "\n")
    (out / "stats.json").write_text(json.dumps({"teacher": cfg["teacher"], **gen, **stats}, indent=1))
    print(json.dumps(stats, indent=1))

    if args.push:
        from distill_lab.data.prepare import push

        push({"gsm8k_train": kept}, args.push)
        print(f"pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
