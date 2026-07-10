# Muster

Multi-agent SOC PoC: detect a prompt-injection-hijacked teammate via in-context
peer behavior modeling (dual of ad-hoc teamwork / CooT). See README.md for the
threat model and architecture.

## Structure

- `src/muster/` — library. Module map: ① `injection.py` ② `teammate_model.py`
  ③ `scoring.py` ④ `defense.py`; environment in `state.py` / `agents.py` /
  `graph.py`; detectors + metrics in `baselines.py` / `evaluate.py`.
- `scripts/` — thin wrappers over `muster.cli` (use `_bootstrap.py` for path setup).
- `data/` — generated datasets/plots; gitignored except `.gitkeep`.

## Commands

```bash
.venv/bin/python -m pytest -q                     # tests (pythonpath handled by pyproject)
.venv/bin/python scripts/run_scenario.py --inject # single-incident demo
.venv/bin/python scripts/generate_dataset.py && .venv/bin/python scripts/evaluate.py
```

## Conventions

- Default backend is `rule` (deterministic, offline). `llm` backend needs
  `ANTHROPIC_API_KEY` (Claude) or `GEMINI_API_KEY` (Gemini); provider selection
  lives in `llm.try_build_client`. Keep rule/llm behind the same interface in
  `teammate_model.py`.
- Never hardcode detection thresholds — calibrate from normal-trajectory quantiles
  (`defense.py`).
- Detectors implement the shared `fit`/score interface in `baselines.py`; new
  baselines go there so `evaluate.py` picks them up automatically.
- The Responder's intended action must be scored *before* commit (prevention, not
  logging) — preserve this ordering in `graph.py` when editing the graph.
