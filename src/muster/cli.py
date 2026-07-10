"""Command-line entry points.

    muster-scenario   run one incident and print the trajectory + Sentinel verdict
    muster-dataset    generate a balanced normal/attack dataset to JSONL
    muster-evaluate   score all detectors and print the ablation + stealth curve
"""

from __future__ import annotations

import argparse
import json
from typing import Optional

from .datagen import build_dataset, generate_incident, load_dataset, make_email, save_dataset
from .evaluate import evaluate, format_stealth_curve, format_table
from .graph import run_incident
from .injection import embed_payload
from .state import RunConfig


def scenario_main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Run one SOC incident through Muster.")
    p.add_argument("--inject", action="store_true", help="hijack the target agent")
    p.add_argument("--target", default="responder", choices=["analyzer", "responder", "reporter"])
    p.add_argument("--stealth", type=float, default=0.0)
    p.add_argument("--backend", default="rule", choices=["rule", "llm"])
    p.add_argument("--no-defense", action="store_true", help="disable challenge/freeze")
    p.add_argument("--threshold", type=float, default=3.0)
    p.add_argument("--vote", action="store_true")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args(argv)

    import random

    rng = random.Random(args.seed)
    risk = "high" if args.inject else rng.choice(["low", "medium", "high"])
    email = make_email(rng, risk)
    if args.inject:
        email = embed_payload(email)

    config = RunConfig(
        inject=args.inject,
        target=args.target,
        stealth=args.stealth,
        backend=args.backend,
        defense=not args.no_defense,
        threshold=args.threshold,
        vote=args.vote,
        seed=args.seed,
    )
    final = run_incident(email, config)

    print(f"# Incident (risk={risk}, inject={args.inject}, target={args.target}, "
          f"stealth={args.stealth}, defense={config.defense})\n")
    for i, (step, dev) in enumerate(zip(final["trajectory"], final["deviations"])):
        flag = "  <-- ALERT" if dev.score >= config.threshold else ""
        print(f"[{i}] {step.agent}: action={step.action!r} "
              f"surprisal={dev.action_surprisal:.2f} anomaly={dev.rationale_anomaly:.2f} "
              f"score={dev.score:.2f}{flag}")
        print(f"    rationale: {step.rationale}")
    print()
    print(f"committed_action : {final.get('committed_action')}")
    print(f"prevented        : {final.get('prevented')}")
    if final.get("alerts"):
        print("\nsentinel log:")
        for a in final["alerts"]:
            print(f"  - {a}")
    return 0


def dataset_main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate a Muster trajectory dataset.")
    p.add_argument("--out", default="data/dataset.jsonl")
    p.add_argument("--n-per-cell", type=int, default=40)
    p.add_argument("--target", default="responder", choices=["analyzer", "responder", "reporter"])
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    records = build_dataset(n_per_cell=args.n_per_cell, target=args.target, seed=args.seed)
    save_dataset(records, args.out)
    n_atk = sum(1 for r in records if r["is_attack"])
    print(f"Wrote {len(records)} incidents ({n_atk} attacks, {len(records)-n_atk} normal) to {args.out}")
    return 0


def evaluate_main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Evaluate detectors on a Muster dataset.")
    p.add_argument("--data", default="data/dataset.jsonl")
    p.add_argument("--quantile", type=float, default=0.95, help="threshold calibration quantile")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--json", action="store_true", help="also print raw results as JSON")
    args = p.parse_args(argv)

    records = load_dataset(args.data)
    results = evaluate(records, quantile=args.quantile, seed=args.seed)
    print(f"# Ablation ({results['n_fit']} fit / {results['n_eval']} eval incidents)\n")
    print(format_table(results))
    print("\n# Recall vs stealth\n")
    print(format_stealth_curve(results))
    if args.json:
        print("\n" + json.dumps(results, indent=2))
    return 0
