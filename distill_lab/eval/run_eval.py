"""generate answers for every test question and score them.

    # on kaggle (gpu, vllm)
    python -m distill_lab.eval.run_eval --model Qwen/Qwen3-0.6B-Base --name b0
    # on the laptop, a few questions through plain transformers, just to see it work
    python -m distill_lab.eval.run_eval --backend hf --limit 3 --name smoke

trailing key=value args override the config, e.g. eval.max_new_tokens=256
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

from distill_lab.config import load_config
from distill_lab.data.prepare import few_shot_prompt, format_prompt, load_split, shots
from distill_lab.eval.extract import extract_answer, is_correct

# base models finish an answer and start inventing the next problem
STOP = ["Question:"]

Generate = Callable[[list[str]], list[str]]


def build_prompts(records: list[dict], examples: list[dict]) -> list[str]:
    if examples:
        return [few_shot_prompt(r["question"], examples) for r in records]
    return [format_prompt(r["question"]) for r in records]


def truncate_at_stop(text: str, stops: list[str] = STOP) -> str:
    cuts = [i for i in (text.find(s) for s in stops) if i >= 0]
    return text[: min(cuts)] if cuts else text


def score(records: list[dict], outputs: list[str]) -> tuple[list[dict], dict]:
    rows = []
    for r, out in zip(records, outputs, strict=True):
        pred = extract_answer(out)
        rows.append({"id": r["id"], "gold": r["answer"], "pred": pred, "correct": is_correct(pred, r["answer"]), "output": out})
    n = len(rows)
    acc = sum(r["correct"] for r in rows) / n if n else 0.0
    summary = {
        "n": n,
        "accuracy": acc,
        # binomial 95% interval, the "how big a difference is real" number
        "ci95": 1.96 * math.sqrt(acc * (1 - acc) / n) if n else 0.0,
        "no_answer": sum(r["pred"] is None for r in rows),
    }
    return rows, summary


def vllm_generator(model: str, max_new_tokens: int, temperature: float, seed: int) -> Generate:
    from vllm import LLM, SamplingParams

    # t4 has no bf16, so half
    llm = LLM(model=model, dtype="half", seed=seed, max_model_len=4096, gpu_memory_utilization=0.9)
    params = SamplingParams(temperature=temperature, max_tokens=max_new_tokens, stop=STOP, seed=seed)

    def generate(prompts: list[str]) -> list[str]:
        return [o.outputs[0].text for o in llm.generate(prompts, params)]

    return generate


def hf_generator(model: str, max_new_tokens: int, temperature: float, seed: int) -> Generate:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(seed)
    tok = AutoTokenizer.from_pretrained(model)
    lm = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()

    @torch.no_grad()
    def generate(prompts: list[str]) -> list[str]:
        outs = []
        for p in prompts:
            ids = tok(p, return_tensors="pt")["input_ids"]
            kw = {"do_sample": True, "temperature": temperature} if temperature > 0 else {"do_sample": False}
            gen = lm.generate(ids, max_new_tokens=max_new_tokens, stop_strings=STOP, tokenizer=tok, **kw)
            outs.append(truncate_at_stop(tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)))
        return outs

    return generate


def evaluate(datasets: dict[str, list[dict]], examples: list[dict], generate: Generate, out_dir: Path) -> dict[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for name, records in datasets.items():
        t0 = time.time()
        rows, summary = score(records, generate(build_prompts(records, examples)))
        summary["seconds"] = round(time.time() - t0, 1)
        with open(out_dir / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        summaries[name] = summary
    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=1))
    return summaries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--model", default=None, help="hf id or local path, defaults to the config's student")
    ap.add_argument("--name", required=True, help="run name, results go to <out>/<name>/")
    ap.add_argument("--shots", type=int, default=None, help="worked examples in the prompt, defaults to eval.n_shot")
    ap.add_argument("--data", default="cache/data")
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--limit", type=int, default=None, help="only the first n questions of each set")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)
    ev = cfg["eval"]
    model = args.model or cfg["student"]
    n_shot = ev["n_shot"] if args.shots is None else args.shots

    datasets = {f"{d}_test": load_split(args.data, f"{d}_test")[: args.limit] for d in cfg["data"]["eval"]}
    examples = shots(load_split(args.data, "gsm8k_train"))[:n_shot]
    make = vllm_generator if args.backend == "vllm" else hf_generator
    generate = make(model, ev["max_new_tokens"], ev["temperature"], cfg["seed"])

    summaries = evaluate(datasets, examples, generate, Path(args.out) / args.name)
    for name, s in summaries.items():
        print(f"{name:12s} acc {s['accuracy']:.1%} ± {s['ci95']:.1%}  (n={s['n']}, no answer {s['no_answer']}, {s['seconds']}s)")

    if args.wandb:
        import wandb

        run = wandb.init(project=cfg["wandb"]["project"], name=args.name, job_type="eval",
                         config={"model": model, "n_shot": n_shot, **cfg})
        run.log({f"{k}/{m}": v for k, s in summaries.items() for m, v in s.items()})
        run.finish()


if __name__ == "__main__":
    main()
