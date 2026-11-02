from pathlib import Path

import yaml


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _set(cfg: dict, dotted: str, value: str) -> None:
    *path, last = dotted.split(".")
    node = cfg
    for k in path:
        node = node.setdefault(k, {})
    v = yaml.safe_load(value)
    if isinstance(v, str):
        # pyyaml reads "2e-5" as a string, it wants "2.0e-5"
        try:
            v = float(v)
        except ValueError:
            pass
    node[last] = v


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """load a yaml config, following `base:` up the chain, then apply key=value overrides"""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("base", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    for o in overrides or []:
        key, _, value = o.partition("=")
        _set(cfg, key, value)
    return cfg
