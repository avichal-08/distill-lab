import pytest
import torch

from distill_lab.data.tokens import tokenize_example, tokenize_record
from distill_lab.teacher.logits import ShardWriter, build_cache, load_topk, topk_rows
from distill_lab.train.losses import IGNORE

V = 64


class CharTok:
    """one token per character, enough to test alignment without downloading a tokenizer"""

    eos_token_id = 1
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2 + ord(c) % (V - 2) for c in text]}


def tiny_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=V, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=256)
    return Qwen3ForCausalLM(cfg).eval()


def test_tokenize_masks_the_prompt_and_appends_eos():
    ex = tokenize_example(CharTok(), "Q:", " ab", max_len=100)
    assert len(ex["input_ids"]) == 2 + 3 + 1
    assert ex["labels"][:2] == [IGNORE, IGNORE]
    assert ex["labels"][2:] == ex["input_ids"][2:]
    assert ex["input_ids"][-1] == CharTok.eos_token_id


def test_tokenize_truncates_both_the_same_way():
    ex = tokenize_example(CharTok(), "Q:", " abcdef", max_len=4)
    assert len(ex["input_ids"]) == len(ex["labels"]) == 4


def test_topk_rows_one_row_per_label_in_order():
    labels = torch.tensor([IGNORE, IGNORE, 5, 6, 7])
    logits = torch.full((5, 10), -10.0)
    for t, nxt in enumerate(labels[1:].tolist()):
        if nxt != IGNORE:
            logits[t, nxt] = 10.0  # position t predicts the token at t+1
    lp, ids = topk_rows(logits, labels, k=3)
    assert ids.shape == (3, 3) and lp.dtype == torch.float16 and ids.dtype == torch.int32
    assert ids[:, 0].tolist() == [5, 6, 7]


def test_shards_round_trip(tmp_path):
    w = ShardWriter(tmp_path, shard_size=2)
    rows = {f"ex{i}": (torch.randn(i + 1, 4).half(), torch.randint(0, V, (i + 1, 4)).int()) for i in range(5)}
    for ex, (lp, ids) in rows.items():
        w.add(ex, lp, ids)
    w.flush()
    assert w.n_shards == 3
    back = load_topk(tmp_path)
    assert set(back) == set(rows)
    for ex, (lp, ids) in rows.items():
        assert torch.equal(back[ex][0], lp) and torch.equal(back[ex][1], ids)


def test_cache_matches_an_unbatched_forward_pass(tmp_path):
    # padding in a batch must not change any example's rows
    model, tok = tiny_model(), CharTok()
    recs = [{"id": f"r{i}", "question": "x" * (3 + 4 * i), "solution": "y" * (2 + 3 * i)} for i in range(4)]
    stats = build_cache(model, tok, recs, max_len=128, k=8, batch_size=3, writer=ShardWriter(tmp_path, 10))
    cached = load_topk(tmp_path)
    for r in recs:
        ex = tokenize_record(tok, r, 128)
        with torch.no_grad():
            logits = model(input_ids=torch.tensor([ex["input_ids"]])).logits[0]
        lp, ids = topk_rows(logits, torch.tensor(ex["labels"]), 8)
        n_labels = sum(l != IGNORE for l in ex["labels"][1:])
        assert len(cached[r["id"]][0]) == n_labels
        assert torch.equal(cached[r["id"]][1], ids)
        assert torch.allclose(cached[r["id"]][0].float(), lp.float(), atol=1e-2)
    assert stats["examples"] == 4 and 0 < stats["top_k_mass"] <= 1.0 + 1e-3
