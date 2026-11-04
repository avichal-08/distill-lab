"""fine-tune the student with one of the losses.

    python -m distill_lab.train.trainer --config configs/r1.yaml --name r1 --push avichal-08/distill-lab-r1
    python -m distill_lab.train.trainer --config configs/l1.yaml --name l1-1k data.n_questions=1000

trailing key=value args override the config.
"""

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Callable

import torch

from distill_lab.config import load_config
from distill_lab.train.data import build_examples, collate, label_tokens, load_records, load_teacher_topk
from distill_lab.train.losses import TopK
from distill_lab.train.objective import objective


def make_optimizer(model, train_cfg: dict, device: str):
    params = [p for p in model.parameters() if p.requires_grad]
    if train_cfg["optimizer"] == "adamw8bit" and device == "cuda":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])
    return torch.optim.AdamW(params, lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])


def warmup_cosine(total_steps: int, warmup_ratio: float) -> Callable[[int], float]:
    warmup = max(1, int(total_steps * warmup_ratio))

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return factor


def train(cfg: dict, model, tok, examples: list[dict], log: Callable[[dict], None] = print) -> list[dict]:
    tc = cfg["train"]
    device = next(model.parameters()).device.type
    use_fp16 = tc["fp16"] and device == "cuda"
    if tc["grad_ckpt"]:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    window = tc["micro_batch"] * tc["grad_accum"]
    steps_per_epoch = math.ceil(len(examples) / window)
    total = steps_per_epoch * tc["epochs"]
    opt = make_optimizer(model, tc, device)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine(total, tc["warmup_ratio"]))
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rng = random.Random(cfg["seed"])

    history, step, t0 = [], 0, time.time()
    for _ in range(tc["epochs"]):
        order = list(range(len(examples)))
        rng.shuffle(order)
        for w in range(0, len(order), window):
            group = [examples[i] for i in order[w : w + window]]
            # normalise by every label token in the window, so short and long micro-batches weigh right
            n_tok = label_tokens(group)
            parts_sum: dict[str, float] = {}
            for m in range(0, len(group), tc["micro_batch"]):
                b = collate(group[m : m + tc["micro_batch"]], pad)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_fp16):
                    logits = model(input_ids=b["input_ids"].to(device), attention_mask=b["attention_mask"].to(device)).logits
                teacher = b.get("teacher")
                if teacher is not None:
                    teacher = TopK(ids=teacher.ids.to(device), logprobs=teacher.logprobs.to(device))
                loss, parts = objective(cfg["loss"], logits, b["labels"].to(device), tok.eos_token_id, teacher)
                scaler.scale(loss / n_tok).backward()
                for k, v in parts.items():
                    parts_sum[k] = parts_sum.get(k, 0.0) + v
                del logits, loss

            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc["max_grad_norm"]).item()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1

            row = {"step": step, "lr": sched.get_last_lr()[0], "grad_norm": gnorm, "tokens": n_tok,
                   **{k: v / n_tok for k, v in parts_sum.items()}}
            history.append(row)
            if step % tc["log_every"] == 0 or step == total:
                log({**row, "total_steps": total, "elapsed_s": round(time.time() - t0)})
    return history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data", default="cache/data")
    ap.add_argument("--out", default="outputs/train")
    ap.add_argument("--push", default=None, help="hf model repo for the trained student")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(args.config, args.overrides)
    torch.manual_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["student"])
    # fp32 master weights, autocast does the fp16 math
    model = AutoModelForCausalLM.from_pretrained(cfg["student"], dtype=torch.float32).to(device)

    recs = load_records(cfg["data"], args.data, cfg["seed"])
    topk = load_teacher_topk(cfg["data"], args.data) if cfg["loss"]["kind"] != "ce" else None
    examples = build_examples(tok, recs, cfg["data"]["max_len"], topk)
    del topk
    print(f"{len(examples)} examples from {len({r['question_id'] for r in recs})} questions, "
          f"{label_tokens(examples)} label tokens")

    log = print
    if args.wandb:
        import wandb

        run = wandb.init(project=cfg["wandb"]["project"], name=args.name, job_type="train", config=cfg)

        def log(row):
            print(row)
            run.log(row, step=row["step"])

    history = train(cfg, model, tok, examples, log)

    out = Path(args.out) / args.name
    model.config.use_cache = True
    # training is over, fp16 halves the checkpoint and it's what vllm loads anyway
    model.to(torch.float16).save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    (out / "train_history.json").write_text(json.dumps(history))
    (out / "run_config.json").write_text(json.dumps(cfg, indent=1))
    print(f"saved to {out}")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, private=True, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push)
        print(f"pushed to https://huggingface.co/{args.push}")
    if args.wandb:
        run.finish()


if __name__ == "__main__":
    main()
