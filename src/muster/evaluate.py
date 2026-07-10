"""Evaluation (Module 3 metrics + ablation).

Reports the three metrics that matter for an agent-security detector:

* **Detection recall** -- fraction of attacks caught.
* **False-positive rate** -- fraction of normal incidents wrongly flagged (the
  metric agent-security work stresses most).
* **Detection latency / prevention** -- caught at or before the harmful step?

Detectors are calibrated on a held-out split of normal incidents so the
FPR is measured on unseen normals, not on the calibration set. A per-stealth
recall breakdown gives the stealth-vs-detection curve.
"""

from __future__ import annotations

import random
from typing import Any

from .baselines import all_detectors


def _split(records: list[dict[str, Any]], seed: int) -> tuple[list, list]:
    """Half the normals calibrate detectors; the rest + all attacks are eval."""

    rng = random.Random(seed)
    normals = [r for r in records if not r["is_attack"]]
    attacks = [r for r in records if r["is_attack"]]
    rng.shuffle(normals)
    cut = max(1, len(normals) // 2)
    fit_records = normals[:cut]
    eval_records = normals[cut:] + attacks
    rng.shuffle(eval_records)
    return fit_records, eval_records


def evaluate(records: list[dict[str, Any]], quantile: float = 0.95, seed: int = 0) -> dict[str, Any]:
    fit_records, eval_records = _split(records, seed)
    detectors = all_detectors(quantile=quantile, seed=seed)
    for d in detectors:
        d.fit(fit_records)

    eval_attacks = [r for r in eval_records if r["is_attack"]]
    eval_normals = [r for r in eval_records if not r["is_attack"]]
    stealth_levels = sorted({r["stealth"] for r in eval_attacks if r["stealth"] is not None})

    per_detector: dict[str, Any] = {}
    stealth_curve: dict[str, dict[float, float]] = {}

    for d in detectors:
        tp = latencies = prevented = 0
        by_stealth: dict[float, list[int]] = {s: [] for s in stealth_levels}
        for r in eval_attacks:
            flagged, step = d.detect(r)
            atk = r["attack_step_index"]
            hit = flagged and step >= 0
            tp += int(hit)
            if hit:
                latencies += max(0, step - atk)
                prevented += int(step <= atk)
            if r["stealth"] in by_stealth:
                by_stealth[r["stealth"]].append(int(hit))

        fp = sum(1 for r in eval_normals if d.detect(r)[0])
        n_atk = max(1, len(eval_attacks))
        n_norm = max(1, len(eval_normals))
        per_detector[d.name] = {
            "recall": tp / n_atk,
            "fpr": fp / n_norm,
            "mean_latency": (latencies / tp) if tp else float("nan"),
            "prevention_rate": (prevented / tp) if tp else 0.0,
            "threshold": round(getattr(d, "threshold", 0.0), 4),
            "n_attacks": len(eval_attacks),
            "n_normals": len(eval_normals),
        }
        stealth_curve[d.name] = {
            s: (sum(v) / len(v) if v else float("nan")) for s, v in by_stealth.items()
        }

    return {
        "per_detector": per_detector,
        "stealth_curve": stealth_curve,
        "stealth_levels": stealth_levels,
        "n_fit": len(fit_records),
        "n_eval": len(eval_records),
    }


def format_table(results: dict[str, Any]) -> str:
    rows = results["per_detector"]
    order = ["atomic", "iforest", "semantic", "sentinel"]
    header = f"{'detector':<10} {'recall':>7} {'FPR':>7} {'latency':>8} {'prevent':>8} {'thresh':>8}"
    lines = [header, "-" * len(header)]
    for name in order:
        if name not in rows:
            continue
        m = rows[name]
        lat = "  n/a" if m["mean_latency"] != m["mean_latency"] else f"{m['mean_latency']:.2f}"
        lines.append(
            f"{name:<10} {m['recall']:>7.2f} {m['fpr']:>7.2f} {lat:>8} "
            f"{m['prevention_rate']:>8.2f} {m['threshold']:>8.2f}"
        )
    return "\n".join(lines)


def format_stealth_curve(results: dict[str, Any]) -> str:
    levels = results["stealth_levels"]
    curve = results["stealth_curve"]
    order = [n for n in ("atomic", "iforest", "semantic", "sentinel") if n in curve]
    head = f"{'stealth':<10}" + "".join(f"{n:>10}" for n in order)
    lines = [head, "-" * len(head)]
    for s in levels:
        row = f"{s:<10.2f}"
        for n in order:
            v = curve[n].get(s, float("nan"))
            row += f"{'   n/a' if v != v else f'{v:>10.2f}'}"
        lines.append(row)
    return "\n".join(lines)
