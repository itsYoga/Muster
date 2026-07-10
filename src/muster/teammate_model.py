"""Teammate behavior model (Module 2).

Each agent maintains, for every peer, an *independent* estimate of the policy
that peer *should* follow. Detection is the dual of ad-hoc teamwork (CooT): the
same in-context model that lets an agent anticipate a teammate also lets it
notice when the teammate departs from its expected policy.

Two backends implement the same interface:

* :class:`HeuristicTeammateModel` -- a closed-form policy prior. Zero-training,
  deterministic, reproducible; the default used for datasets and evaluation.
* :class:`LLMTeammateModel` -- prompts an LLM (Claude or Gemini, see
  :mod:`muster.llm`) with the peer's trajectory and asks it to predict the next
  action as a distribution; used when ``backend == "llm"``.

Both return a probability distribution over the peer's action space, which the
scorer turns into a surprisal.
"""

from __future__ import annotations

import math
from typing import Optional, Protocol

from .state import ACTIONS, RISK_LEVELS, Email, RiskLevel, Step

# --- Closed-form policy priors -------------------------------------------------

# Expected Responder action given the Analyzer's stated verdict. This encodes
# "a well-behaved responder blocks high-risk domains and only allows low-risk
# ones". The hijacked responder's `allow` on a `high` verdict lands in the tail.
_ACTION_GIVEN_VERDICT: dict[RiskLevel, dict[str, float]] = {
    "high": {"block_domain": 0.85, "quarantine_mailbox": 0.10, "escalate": 0.04, "allow": 0.01},
    "medium": {"quarantine_mailbox": 0.60, "block_domain": 0.20, "escalate": 0.15, "allow": 0.05},
    "low": {"allow": 0.85, "escalate": 0.10, "quarantine_mailbox": 0.04, "block_domain": 0.01},
}


def ioc_risk_score(email: Email) -> float:
    """Independent [0, 1] risk estimate from raw indicators.

    This is the teammate model *of the Analyzer*: it lets a peer judge whether
    the Analyzer's emitted verdict is consistent with the evidence, catching an
    upstream hijack that downgrades the verdict while the IOCs stay malicious.
    """

    s = email.ioc_signals
    url = float(s.get("url_reputation", 0.0))  # P(malicious) of the URL
    sender = float(s.get("sender_reputation", 0.0))  # P(malicious) of the sender
    attach = float(s.get("attachment_malicious", 0.0))  # 0/1 flag
    # Weighted evidence; attachment is the strongest single signal.
    return max(0.0, min(1.0, 0.45 * url + 0.25 * sender + 0.30 * attach))


def expected_verdict_distribution(email: Email, sharpness: float = 6.0) -> dict[str, float]:
    """Distribution over risk levels the Analyzer is expected to emit."""

    risk = ioc_risk_score(email)
    # Centers for low/medium/high on the risk axis.
    centers = {"low": 0.15, "medium": 0.5, "high": 0.85}
    logits = {lvl: -sharpness * (risk - c) ** 2 for lvl, c in centers.items()}
    return _softmax(logits)


def expected_action_distribution(verdict: Optional[str]) -> dict[str, float]:
    """Distribution over Responder actions given the stated verdict."""

    if verdict not in _ACTION_GIVEN_VERDICT:
        # No verdict in context -> uniform prior (maximally uncertain).
        return {a: 1.0 / len(ACTIONS) for a in ACTIONS}
    return dict(_ACTION_GIVEN_VERDICT[verdict])


def expected_report_keywords(verdict: Optional[str], action: Optional[str]) -> set[str]:
    """Content the Reporter is expected to faithfully preserve."""

    kws: set[str] = set()
    if verdict:
        kws.add(verdict)
    if action:
        kws.add(action)
    return kws


class TeammateModel(Protocol):
    """Predicts a peer's next-action distribution from trajectory context."""

    def predict(self, peer: str, trajectory: list[Step], email: Email) -> dict[str, float]:
        ...


class HeuristicTeammateModel:
    """Closed-form teammate model (default)."""

    def predict(self, peer: str, trajectory: list[Step], email: Email) -> dict[str, float]:
        if peer == "analyzer":
            return expected_verdict_distribution(email)
        if peer == "responder":
            verdict = _last_action_of(trajectory, "analyzer")
            return expected_action_distribution(verdict)
        # Reporter is scored by content consistency, not an action distribution.
        return {}


class LLMTeammateModel:
    """Claude-backed teammate model.

    Asks the model to predict the peer's next action as a distribution given the
    peer's observed trajectory. Falls back to the heuristic prior on any error so
    the pipeline never hard-fails when the API is unavailable.
    """

    def __init__(self, client, model: str = "claude-opus-4-8") -> None:
        self._client = client
        self._model = model
        self._fallback = HeuristicTeammateModel()

    def predict(self, peer: str, trajectory: list[Step], email: Email) -> dict[str, float]:
        if peer not in ("analyzer", "responder"):
            return {}
        labels = list(RISK_LEVELS) if peer == "analyzer" else list(ACTIONS)
        try:
            dist = self._client.predict_distribution(peer, trajectory, email, labels)
            return _normalize(dist, labels)
        except Exception:  # pragma: no cover - network/availability guard
            return self._fallback.predict(peer, trajectory, email)


# --- helpers -------------------------------------------------------------------


def surprisal(distribution: dict[str, float], actual: str, floor: float = 1e-4) -> float:
    """``-log P(actual)`` with a probability floor to bound the tail."""

    p = max(float(distribution.get(actual, 0.0)), floor)
    return -math.log(p)


def _softmax(logits: dict[str, float]) -> dict[str, float]:
    m = max(logits.values())
    exps = {k: math.exp(v - m) for k, v in logits.items()}
    z = sum(exps.values())
    return {k: v / z for k, v in exps.items()}


def _normalize(dist: dict[str, float], labels: list[str]) -> dict[str, float]:
    clean = {k: max(0.0, float(dist.get(k, 0.0))) for k in labels}
    z = sum(clean.values())
    if z <= 0:
        return {k: 1.0 / len(labels) for k in labels}
    return {k: v / z for k, v in clean.items()}


def _last_action_of(trajectory: list[Step], agent: str) -> Optional[str]:
    for step in reversed(trajectory):
        if step.agent == agent:
            return step.action
    return None
