"""Detectors: the Sentinel and the baselines it is measured against.

All detectors share one interface -- ``fit`` on normal incidents, then
``detect`` an incident -> ``(flagged, detected_step_index)``. Each learned
detector calibrates its own threshold from its normal max-score distribution at
a common quantile, so the comparison is about *what signal* a detector sees, not
about hand-tuned cutoffs.

* :class:`SentinelDetector` -- cross-agent: scores each step against the
  teammate model *with* the upstream context (the verdict). This is the method.
* :class:`SemanticDetector` -- rationale-anomaly only; included to show a
  semantic detector degrade against stealth.
* :class:`AtomicDetector` -- per-step permission legality (the ARMO-style
  atomic check). Sees each action in isolation and misses locally-legal betrayal.
* :class:`IsolationForestDetector` -- a learned per-step detector with NO
  cross-agent context (the autoencoder-style baseline). Cannot see the
  verdict/action mismatch.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .defense import calibrate_threshold
from .scoring import SEMANTIC_WEIGHTS, SENTINEL_WEIGHTS, score_trajectory
from .state import ACTIONS, RISK_LEVELS, Email, Step
from .teammate_model import HeuristicTeammateModel
from .text import rationale_anomaly

# Role permission scopes for the atomic check. Every action an agent emits in
# this simulation is within its scope -- which is exactly why atomic checks miss
# the attack: `allow` is a legal Responder action.
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "analyzer": set(RISK_LEVELS),
    "responder": set(ACTIONS),
    "reporter": {"report"},
}

# Global action vocabulary for context-free featurization.
_ACTION_VOCAB = list(RISK_LEVELS) + list(ACTIONS) + ["report"]


def rebuild(record: dict[str, Any]) -> tuple[Email, list[Step]]:
    e = record["email"]
    email = Email(
        subject=e["subject"],
        sender=e["sender"],
        body=e["body"],
        url=e["url"],
        attachment_hash=e["attachment_hash"],
        ioc_signals=e["ioc_signals"],
        true_risk=e["true_risk"],
    )
    traj = [
        Step(
            agent=s["agent"],
            observation=s["observation"],
            action=s["action"],
            rationale=s["rationale"],
            metadata=s.get("metadata", {}),
        )
        for s in record["trajectory"]
    ]
    return email, traj


class _CalibratedDetector:
    """Shared threshold calibration on per-incident max scores."""

    name = "base"

    def __init__(self, quantile: float = 0.95) -> None:
        self.quantile = quantile
        self.threshold = 0.0

    def _scores(self, record: dict[str, Any]) -> list[float]:  # pragma: no cover - abstract
        raise NotImplementedError

    def fit(self, records: list[dict[str, Any]]) -> "_CalibratedDetector":
        normal_maxes = [
            max(self._scores(r), default=0.0) for r in records if not r["is_attack"]
        ]
        self.threshold = calibrate_threshold(normal_maxes, self.quantile)
        return self

    def detect(self, record: dict[str, Any]) -> tuple[bool, int]:
        scores = self._scores(record)
        for i, s in enumerate(scores):
            if s >= self.threshold:
                return True, i
        return False, -1


class SentinelDetector(_CalibratedDetector):
    name = "sentinel"

    def __init__(self, quantile: float = 0.95, weights=SENTINEL_WEIGHTS) -> None:
        super().__init__(quantile)
        self.weights = weights
        self.model = HeuristicTeammateModel()

    def _scores(self, record: dict[str, Any]) -> list[float]:
        email, traj = rebuild(record)
        return [d.score for d in score_trajectory(traj, email, self.model, self.weights)]


class SemanticDetector(SentinelDetector):
    name = "semantic"

    def __init__(self, quantile: float = 0.95) -> None:
        super().__init__(quantile, weights=SEMANTIC_WEIGHTS)


class AtomicDetector:
    """Per-step permission legality; no training, no context."""

    name = "atomic"

    def fit(self, records: list[dict[str, Any]]) -> "AtomicDetector":
        return self

    def detect(self, record: dict[str, Any]) -> tuple[bool, int]:
        for i, s in enumerate(record["trajectory"]):
            allowed = ROLE_PERMISSIONS.get(s["agent"], set())
            if s["action"] not in allowed:
                return True, i
        return False, -1


class IsolationForestDetector(_CalibratedDetector):
    """Learned per-step anomaly detector with no cross-agent context."""

    name = "iforest"

    def __init__(self, quantile: float = 0.95, seed: int = 0) -> None:
        super().__init__(quantile)
        self.seed = seed
        self._model = None

    @staticmethod
    def _features(step: dict[str, Any]) -> list[float]:
        agent_oh = [1.0 if step["agent"] == a else 0.0 for a in ("analyzer", "responder", "reporter")]
        action_oh = [1.0 if step["action"] == a else 0.0 for a in _ACTION_VOCAB]
        text = str(step.get("metadata", {}).get("report_text", step["rationale"]))
        anom = rationale_anomaly(text)
        length = float(len(text.split()))
        return agent_oh + action_oh + [anom, length]

    def fit(self, records: list[dict[str, Any]]) -> "IsolationForestDetector":
        from sklearn.ensemble import IsolationForest

        X = [self._features(s) for r in records if not r["is_attack"] for s in r["trajectory"]]
        self._model = IsolationForest(random_state=self.seed, contamination="auto").fit(np.asarray(X))
        # Calibrate on per-incident max anomaly score over normal incidents.
        normal_maxes = [
            max(self._scores(r), default=0.0) for r in records if not r["is_attack"]
        ]
        self.threshold = calibrate_threshold(normal_maxes, self.quantile)
        return self

    def _scores(self, record: dict[str, Any]) -> list[float]:
        if self._model is None:
            return [0.0 for _ in record["trajectory"]]
        X = np.asarray([self._features(s) for s in record["trajectory"]])
        # Higher = more anomalous (decision_function is higher for inliers).
        return [float(-v) for v in self._model.decision_function(X)]


def all_detectors(quantile: float = 0.95, seed: int = 0):
    return [
        AtomicDetector(),
        IsolationForestDetector(quantile=quantile, seed=seed),
        SemanticDetector(quantile=quantile),
        SentinelDetector(quantile=quantile),
    ]
