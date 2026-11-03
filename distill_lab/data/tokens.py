from distill_lab.data.prepare import format_example
from distill_lab.train.losses import IGNORE


def tokenize_example(tok, prompt: str, target: str, max_len: int) -> dict[str, list[int]]:
    """input ids and labels for one training example, prompt positions labelled IGNORE.
    the teacher cache and the trainer both go through here, so their positions line up"""
    # tokenized apart, the way the model meets them at generation time: prompt given, target generated
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    # eos so the student learns to stop after the '#### n' line
    t = tok(target, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return {"input_ids": (p + t)[:max_len], "labels": ([IGNORE] * len(p) + t)[:max_len]}


def tokenize_record(tok, rec: dict, max_len: int) -> dict[str, list[int]]:
    return tokenize_example(tok, *format_example(rec), max_len)
