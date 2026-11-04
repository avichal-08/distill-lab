from pathlib import Path

from distill_lab.config import load_config

BASE = Path(__file__).parent.parent / "configs" / "base.yaml"


def test_base_loads():
    cfg = load_config(BASE)
    assert cfg["student"] == "Qwen/Qwen3-0.6B-Base"
    assert cfg["loss"]["kind"] == "ce"


def test_child_inherits_and_overrides(tmp_path):
    child = tmp_path / "child.yaml"
    child.write_text(f"base: {BASE.as_posix()}\nname: l1\nloss:\n  kind: fkl\n")
    cfg = load_config(child)
    assert cfg["name"] == "l1"
    assert cfg["loss"]["kind"] == "fkl"
    # untouched siblings survive the merge
    assert cfg["loss"]["top_k"] == 32
    assert cfg["train"]["lr"] == 1e-5


def test_cli_overrides_are_typed():
    cfg = load_config(BASE, ["train.lr=2e-5", "data.n_questions=1000", "train.fp16=false"])
    assert cfg["train"]["lr"] == 2e-5
    assert cfg["data"]["n_questions"] == 1000
    assert cfg["train"]["fp16"] is False
