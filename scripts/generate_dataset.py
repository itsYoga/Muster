#!/usr/bin/env python3
"""Generate a balanced normal/attack dataset.

    python scripts/generate_dataset.py --out data/dataset.jsonl --n-per-cell 40
"""

import _bootstrap  # noqa: F401
from muster.cli import dataset_main

if __name__ == "__main__":
    raise SystemExit(dataset_main())
