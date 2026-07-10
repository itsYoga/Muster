#!/usr/bin/env python3
"""Run a single incident. Examples:

    python scripts/run_scenario.py                       # normal incident
    python scripts/run_scenario.py --inject              # hijacked responder
    python scripts/run_scenario.py --inject --stealth 1  # stealthy cover story
    python scripts/run_scenario.py --inject --no-defense # watch the attack succeed
"""

import _bootstrap  # noqa: F401
from muster.cli import scenario_main

if __name__ == "__main__":
    raise SystemExit(scenario_main())
