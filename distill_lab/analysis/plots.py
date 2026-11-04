"""accuracy chart for the readme.

    python -m distill_lab.analysis.plots
"""

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = ["b0", "s1-4shot", "r1", "l1", "l2", "b1"]
COLORS = {"gsm8k": "#2a78d6", "svamp": "#eb6834"}
INK, MUTED, SURFACE = "#2b2b29", "#7a7a75", "#fcfcfb"


def load(path: str | Path = "results/accuracy.csv") -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        return {r["run"]: r for r in csv.DictReader(f)}


def accuracy_chart(rows: dict[str, dict], out: str | Path = "results/accuracy.png") -> Path:
    runs = [rows[r] for r in RUNS]
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    width = 0.38
    for i, (key, name) in enumerate([("gsm8k", "GSM8K (trained on)"), ("svamp", "SVAMP (never seen)")]):
        xs = [j + (i - 0.5) * (width + 0.02) for j in range(len(runs))]
        vals = [float(r[key]) for r in runs]
        cis = [float(r[f"{key}_ci"]) for r in runs]
        ax.bar(xs, vals, width, color=COLORS[key], label=name, zorder=2)
        ax.errorbar(xs, vals, yerr=cis, fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, zorder=3)
        for x, v, c in zip(xs, vals, cis):
            ax.text(x, v + c + 1, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color=INK)

    ax.set_xticks(range(len(runs)))
    ax.set_xticklabels([f"{r['label']}\n{r['prompt']}" for r in runs], fontsize=8, color=INK)
    ax.set_ylabel("accuracy (%)", color=MUTED, fontsize=9)
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.tick_params(axis="y", colors=MUTED, labelsize=8)
    ax.tick_params(axis="x", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0cfc8")
    ax.legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    ax.set_title("Qwen3-0.6B distilled from Qwen3-4B, error bars are 95% intervals",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


def gain_chart(rows: dict[str, dict], out: str | Path = "results/gain.png") -> Path:
    """points gained over the base student, with the 95% interval of the difference"""
    base = rows["b0"]
    runs = [rows[r] for r in ["s1-4shot", "r1", "l1", "l2"]]
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    height = 0.36
    for i, (key, name) in enumerate([("gsm8k", "GSM8K (trained on)"), ("svamp", "SVAMP (never seen)")]):
        ys = [j + (i - 0.5) * (height + 0.02) for j in range(len(runs))]
        gains = [float(r[key]) - float(base[key]) for r in runs]
        # independent test sets per model, so the intervals add in quadrature
        cis = [(float(r[f"{key}_ci"]) ** 2 + float(base[f"{key}_ci"]) ** 2) ** 0.5 for r in runs]
        ax.barh(ys, gains, height, color=COLORS[key], label=name, zorder=2)
        ax.errorbar(gains, ys, xerr=cis, fmt="none", ecolor=MUTED, elinewidth=1, capsize=2, zorder=3)
        for y, g, c in zip(ys, gains, cis):
            x = g + c + 0.4 if g >= 0 else g - c - 0.4
            ax.text(x, y, f"{g:+.1f}", va="center", ha="left" if g >= 0 else "right", fontsize=8, color=INK)

    ax.set_yticks(range(len(runs)))
    ax.set_yticklabels([r["label"] for r in runs], fontsize=9, color=INK)
    ax.invert_yaxis()
    ax.axvline(0, color="#a3a29b", linewidth=1, zorder=1)
    ax.set_xlim(-8, 24)
    ax.set_xlabel("points gained over the base student", color=MUTED, fontsize=9)
    ax.xaxis.grid(True, color="#e6e5e0", linewidth=0.8, zorder=0)
    ax.tick_params(axis="x", colors=MUTED, labelsize=8)
    ax.tick_params(axis="y", length=0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#d0cfc8")
    ax.legend(frameon=False, fontsize=9, loc="upper right", labelcolor=INK)
    ax.set_title("gain over the base student, error bars are 95% intervals of the difference",
                 fontsize=10, color=INK, loc="left")
    fig.tight_layout()
    out = Path(out)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    rows = load()
    print(accuracy_chart(rows))
    print(gain_chart(rows))
