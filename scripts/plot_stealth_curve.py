#!/usr/bin/env python3
"""Plot recall vs. attack stealth for every detector.

    python scripts/plot_stealth_curve.py --data data/dataset.jsonl --out data/stealth_curve.png

Requires matplotlib (``pip install matplotlib``). Degrades to a printed table.
"""

import argparse

import _bootstrap  # noqa: F401
from muster.datagen import load_dataset
from muster.evaluate import evaluate, format_stealth_curve


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/dataset.jsonl")
    p.add_argument("--out", default="data/stealth_curve.png")
    p.add_argument("--quantile", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    results = evaluate(load_dataset(args.data), quantile=args.quantile, seed=args.seed)
    levels = results["stealth_levels"]
    curve = results["stealth_curve"]

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; printing table instead.\n")
        print(format_stealth_curve(results))
        return 0

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    order = [n for n in ("atomic", "iforest", "semantic", "sentinel") if n in curve]
    for name in order:
        ys = [curve[name].get(s, float("nan")) for s in levels]
        ax.plot(levels, ys, marker="o", label=name)
    ax.set_xlabel("attack stealth (0 = blatant cover story, 1 = benign-looking)")
    ax.set_ylabel("detection recall")
    ax.set_title("Recall vs. attack stealth")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
