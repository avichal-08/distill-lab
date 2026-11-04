import random
from pathlib import Path

import torch

from distill_lab.data.prepare import load_split
from distill_lab.data.tokens import tokenize_record
from distill_lab.train.losses import IGNORE, TopK


def load_records(data_cfg: dict, data_dir: str | Path, seed: int) -> list[dict]:
    """training records for the configured source, cut to n_questions if set.
    the cut is by question, so teacher data keeps every kept solution of each chosen question"""
    if data_cfg["source"] == "human":
        recs = [dict(r, question_id=r["id"]) for r in load_split(data_dir, "gsm8k_train")]
    elif data_cfg["source"] == "teacher":
        local = Path(data_dir).parent / "teacher"
        if (local / "gsm8k_train.jsonl").exists():
            recs = load_split(local, "gsm8k_train")
        else:
            from datasets import load_dataset

            recs = list(load_dataset(data_cfg["teacher_repo"], split="gsm8k_train"))
    else:
        raise ValueError(f"unknown data.source {data_cfg['source']!r}")

    n = data_cfg.get("n_questions")
    if n:
        qids = sorted({r["question_id"] for r in recs})
        keep = set(random.Random(seed).sample(qids, min(n, len(qids))))
        recs = [r for r in recs if r["question_id"] in keep]
    return recs


def load_teacher_topk(data_cfg: dict, data_dir: str | Path) -> dict:
    from distill_lab.teacher.logits import load_topk

    local = Path(data_dir).parent / "logits"
    if any(local.glob("shard_*.safetensors")):
        return load_topk(local)
    from huggingface_hub import snapshot_download

    return load_topk(snapshot_download(data_cfg["logits_repo"], repo_type="dataset"))


def build_examples(tok, recs: list[dict], max_len: int, topk: dict | None = None) -> list[dict]:
    out = []
    for r in recs:
        ex = tokenize_record(tok, r, max_len)
        ex["id"] = r["id"]
        if topk is not None:
            lp, ids = topk[r["id"]]
            n_labels = sum(l != IGNORE for l in ex["labels"][1:])
            # one cached row per graded position, or the teacher was run on different tokens
            if len(lp) != n_labels:
                raise ValueError(f"{r['id']}: {len(lp)} teacher rows for {n_labels} labels")
            ex["topk"] = (lp, ids)
        out.append(ex)
    return out


def collate(batch: list[dict], pad_id: int) -> dict:
    width = max(len(e["input_ids"]) for e in batch)
    n = len(batch)
    input_ids = torch.full((n, width), pad_id, dtype=torch.long)
    labels = torch.full((n, width), IGNORE, dtype=torch.long)
    attn = torch.zeros((n, width), dtype=torch.long)
    for i, e in enumerate(batch):
        L = len(e["input_ids"])
        input_ids[i, :L] = torch.tensor(e["input_ids"])
        labels[i, :L] = torch.tensor(e["labels"])
        attn[i, :L] = 1
    out = {"input_ids": input_ids, "labels": labels, "attention_mask": attn}

    if "topk" in batch[0]:
        k = batch[0]["topk"][0].shape[-1]
        # zeros where there's no label, those positions are masked out of every loss
        lp = torch.zeros((n, width - 1, k), dtype=torch.float16)
        ids = torch.zeros((n, width - 1, k), dtype=torch.long)
        graded = labels[:, 1:] != IGNORE
        for i, e in enumerate(batch):
            lp[i, graded[i]] = e["topk"][0]
            ids[i, graded[i]] = e["topk"][1].long()
        out["teacher"] = TopK(ids=ids, logprobs=lp)
    return out


def label_tokens(batch: list[dict]) -> int:
    return sum(sum(l != IGNORE for l in e["labels"][1:]) for e in batch)
