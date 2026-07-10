#!/usr/bin/env python3
"""Score all detectors on a dataset and print the ablation + stealth curve.

    python scripts/evaluate.py --data data/dataset.jsonl
"""

import _bootstrap  # noqa: F401
from muster.cli import evaluate_main

if __name__ == "__main__":
    raise SystemExit(evaluate_main())
