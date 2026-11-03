"""how much of the teacher's distribution does a top-k cache keep, and how much
does that move the kd losses.

    python -m distill_lab.analysis.topk_coverage --model Qwen/Qwen3-0.6B-Base --n 20
    python -m distill_lab.analysis.topk_coverage --model Qwen/Qwen3-4B --student Qwen/Qwen3-0.6B-Base
"""

import argparse

import torch
import torch.nn.functional as F

from distill_lab.train.losses import IGNORE, TopK, fkl_loss, jsd_loss, rkl_loss


def topk_stats(
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    ks: list[int],
    student_logits: torch.Tensor | None = None,
) -> dict[int, dict[str, float]]:
    logp = F.log_softmax(teacher_logits.float(), dim=-1)
    top_lp, top_ids = logp.topk(max(ks), dim=-1)
    labels = torch.where(mask, 0, IGNORE)
    dense = None
    if student_logits is not None:
        dense = {f.__name__: f(student_logits, teacher_logits, labels).item() for f in (fkl_loss, rkl_loss, jsd_loss)}

    out = {}
    for k in ks:
        mass = top_lp[..., :k].exp().sum(-1)[mask]
        row = {
            "mass_mean": mass.mean().item(),
            "mass_p10": mass.quantile(0.1).item(),
            # positions where the cache throws away more than 1% of the teacher
            "frac_tail_over_1pct": (mass < 0.99).float().mean().item(),
        }
        if dense is not None:
            cache = TopK(ids=top_ids[..., :k], logprobs=top_lp[..., :k])
            for f in (fkl_loss, rkl_loss, jsd_loss):
                approx = f(student_logits, cache, labels).item()
                row[f"{f.__name__}_rel_err"] = (approx - dense[f.__name__]) / max(dense[f.__name__], 1e-12)
        out[k] = row
    return out


def _load(name: str, device: str):
    from transformers import AutoModelForCausalLM

    dtype = torch.float16 if device == "cuda" else torch.float32
    return AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()


def _examples(tok, n: int, max_len: int):
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n}]")
    for ex in ds:
        prompt = f"Question: {ex['question']}\nAnswer:"
        n_prompt = len(tok(prompt)["input_ids"])
        ids = tok(prompt + " " + ex["answer"], return_tensors="pt")["input_ids"][:, :max_len]
        mask = torch.zeros_like(ids, dtype=torch.bool)
        # only score positions that predict answer tokens
        mask[:, n_prompt - 1 : -1] = True
        yield ids, mask


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base", help="the teacher whose top-k gets cached")
    ap.add_argument("--student", default=None, help="if given, also measure kd loss error")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--ks", default="8,16,32,64")
    ap.add_argument("--max-len", type=int, default=512)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ks = [int(k) for k in args.ks.split(",")]
    tok = AutoTokenizer.from_pretrained(args.model)
    teacher = _load(args.model, device)
    student = _load(args.student, device) if args.student else None

    rows: dict[int, list[dict[str, float]]] = {k: [] for k in ks}
    for ids, mask in _examples(tok, args.n, args.max_len):
        ids, mask = ids.to(device), mask.to(device)
        t = teacher(ids).logits
        s = student(ids).logits if student is not None else None
        for k, row in topk_stats(t, mask, ks, s).items():
            rows[k].append(row)

    print(f"teacher {args.model}, student {args.student}, {args.n} gsm8k solutions")
    keys = list(rows[ks[0]][0])
    print("k".rjust(4) + "".join(key.rjust(22) for key in keys))
    for k in ks:
        avg = {key: sum(r[key] for r in rows[k]) / len(rows[k]) for key in keys}
        print(str(k).rjust(4) + "".join(f"{avg[key]:22.4f}" for key in keys))


if __name__ == "__main__":
    main()
