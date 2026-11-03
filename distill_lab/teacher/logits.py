"""run the teacher over its own kept solutions and save its top-k log-probs at every solution token.

    python -m distill_lab.teacher.logits --solutions cache/teacher/gsm8k_train.jsonl --out cache/logits
    python -m distill_lab.teacher.logits ... --push <hf-user>/distill-lab-logits

rows for one example are in label order: row i is the teacher's prediction for the i-th
non-IGNORE label after the usual shift, which is exactly the order the trainer needs.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from distill_lab.config import load_config
from distill_lab.data.prepare import load_split
from distill_lab.data.tokens import tokenize_record
from distill_lab.train.losses import IGNORE


def topk_rows(logits: torch.Tensor, labels: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, V) logits and (T,) unshifted labels of one example -> (n, k) fp16 log-probs and int32 ids,
    one row per position whose next token is a label"""
    lg, lb = logits[:-1], labels[1:]
    lp = F.log_softmax(lg[lb != IGNORE].float(), dim=-1)
    top = lp.topk(k, dim=-1)
    return top.values.half(), top.indices.int()


class ShardWriter:
    def __init__(self, out: Path, shard_size: int):
        self.out, self.shard_size = Path(out), shard_size
        self.out.mkdir(parents=True, exist_ok=True)
        self._ids, self._lp, self._tok, self.n_shards = [], [], [], 0

    def add(self, example_id: str, logprobs: torch.Tensor, ids: torch.Tensor) -> None:
        self._ids.append(example_id)
        self._lp.append(logprobs.cpu())
        self._tok.append(ids.cpu())
        if len(self._ids) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self._ids:
            return
        lengths = torch.tensor([0] + [len(x) for x in self._lp])
        save_file(
            {"logprobs": torch.cat(self._lp), "ids": torch.cat(self._tok), "offsets": lengths.cumsum(0)},
            str(self.out / f"shard_{self.n_shards:04d}.safetensors"),
            metadata={"example_ids": json.dumps(self._ids)},
        )
        self.n_shards += 1
        self._ids, self._lp, self._tok = [], [], []


def load_topk(path: str | Path) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """example id -> ((n, k) fp16 log-probs, (n, k) int32 ids)"""
    out = {}
    for shard in sorted(Path(path).glob("shard_*.safetensors")):
        with safe_open(str(shard), framework="pt") as f:
            ids = json.loads(f.metadata()["example_ids"])
        t = load_file(str(shard))
        o = t["offsets"].tolist()
        for i, ex in enumerate(ids):
            out[ex] = (t["logprobs"][o[i] : o[i + 1]], t["ids"][o[i] : o[i + 1]])
    return out


@torch.no_grad()
def build_cache(model, tok, records: list[dict], max_len: int, k: int, batch_size: int, writer: ShardWriter) -> dict:
    device = next(model.parameters()).device
    examples = [(r["id"], tokenize_record(tok, r, max_len)) for r in records]
    # similar lengths together, so batches waste little on padding
    examples.sort(key=lambda e: len(e[1]["input_ids"]))
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    n_rows, mass, agree = 0, 0.0, 0
    for b in range(0, len(examples), batch_size):
        batch = examples[b : b + batch_size]
        width = max(len(e["input_ids"]) for _, e in batch)
        ids = torch.full((len(batch), width), pad, dtype=torch.long)
        attn = torch.zeros((len(batch), width), dtype=torch.long)
        for i, (_, e) in enumerate(batch):
            ids[i, : len(e["input_ids"])] = torch.tensor(e["input_ids"])
            attn[i, : len(e["input_ids"])] = 1
        logits = model(input_ids=ids.to(device), attention_mask=attn.to(device)).logits
        for i, (ex_id, e) in enumerate(batch):
            n = len(e["input_ids"])
            labels = torch.tensor(e["labels"], device=device)
            lp, top_ids = topk_rows(logits[i, :n], labels, k)
            writer.add(ex_id, lp, top_ids)
            nxt = labels[1:][labels[1:] != IGNORE]
            n_rows += len(lp)
            mass += lp.float().exp().sum(-1).sum().item()
            agree += (top_ids[:, 0] == nxt).sum().item()
    writer.flush()
    return {"examples": len(examples), "rows": n_rows, "top_k_mass": mass / max(n_rows, 1),
            "teacher_top1_matches_text": agree / max(n_rows, 1), "shards": writer.n_shards}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--solutions", default="cache/teacher/gsm8k_train.jsonl")
    ap.add_argument("--out", default="cache/logits")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--push", default=None, help="hf dataset repo to upload the shards to")
    ap.add_argument("overrides", nargs="*")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = load_config(args.config, args.overrides)
    path = Path(args.solutions)
    records = load_split(path.parent, path.stem)[: args.limit]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["teacher"])
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(cfg["teacher"], dtype=dtype).to(device).eval()

    writer = ShardWriter(Path(args.out), args.shard_size)
    stats = build_cache(model, tok, records, cfg["data"]["max_len"], cfg["loss"]["top_k"], args.batch_size, writer)
    stats.update(teacher=cfg["teacher"], top_k=cfg["loss"]["top_k"], max_len=cfg["data"]["max_len"])
    (Path(args.out) / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push, repo_type="dataset", private=True, exist_ok=True)
        api.upload_folder(folder_path=args.out, repo_id=args.push, repo_type="dataset")
        print(f"pushed to https://huggingface.co/datasets/{args.push}")


if __name__ == "__main__":
    main()
