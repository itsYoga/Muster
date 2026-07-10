"""Core domain types for the Muster SOC simulation.

The scenario is a phishing-email incident-response pipeline handled by three
heterogeneous agents (Analyzer -> Responder -> Reporter). Every agent action is
recorded as a :class:`Step` in a shared, observable trajectory. Detection works
off that trajectory, never off privileged internal state, which is what makes it
a realistic monitor for an agentic system.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, Optional, TypedDict

# --- Enumerations (kept as plain strings so they serialize cleanly to JSONL) ---

RiskLevel = Literal["low", "medium", "high"]
RISK_LEVELS: tuple[RiskLevel, ...] = ("low", "medium", "high")

Action = Literal["allow", "quarantine_mailbox", "block_domain", "escalate"]
ACTIONS: tuple[Action, ...] = ("allow", "quarantine_mailbox", "block_domain", "escalate")

AgentRole = Literal["analyzer", "responder", "reporter"]
AGENT_ROLES: tuple[AgentRole, ...] = ("analyzer", "responder", "reporter")


@dataclass
class Email:
    """A phishing-triage work item plus its latent ground-truth risk.

    ``ioc_signals`` are the observable indicators the Analyzer reasons over.
    ``true_risk`` is the label used only for dataset construction and metrics.
    """

    subject: str
    sender: str
    body: str
    url: str
    attachment_hash: str
    ioc_signals: dict[str, float]  # url_reputation, sender_reputation, attachment_malicious
    true_risk: RiskLevel

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "sender": self.sender,
            "body": self.body,
            "url": self.url,
            "attachment_hash": self.attachment_hash,
            "ioc_signals": self.ioc_signals,
            "true_risk": self.true_risk,
        }


@dataclass
class Step:
    """One observable (observation, action, rationale) tuple in the trajectory."""

    agent: AgentRole
    observation: str
    action: str  # a RiskLevel for the analyzer, an Action for the responder, etc.
    rationale: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "observation": self.observation,
            "action": self.action,
            "rationale": self.rationale,
            "metadata": self.metadata,
        }


@dataclass
class Deviation:
    """The Sentinel's per-step assessment of a peer.

    ``action_surprisal`` is the structural (relational) signal:
    ``-log P(actual_action | context)`` under the teammate model. It cannot be
    hidden by dressing up the rationale. ``rationale_anomaly`` is the semantic
    signal, which a stealthy attacker *can* suppress -- included so we can show
    a semantic-only detector degrade against the robust structural one.
    """

    agent: AgentRole
    step_index: int
    action_surprisal: float
    rationale_anomaly: float
    score: float
    actual: str
    expected: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "step_index": self.step_index,
            "action_surprisal": round(self.action_surprisal, 4),
            "rationale_anomaly": round(self.rationale_anomaly, 4),
            "score": round(self.score, 4),
            "actual": self.actual,
            "expected": {k: round(v, 4) for k, v in self.expected.items()},
        }


@dataclass
class RunConfig:
    """Configuration for a single incident run."""

    inject: bool = False
    target: AgentRole = "responder"  # which single agent is hijacked, if inject
    stealth: float = 0.0  # 0.0 = blatant cover story, 1.0 = mimics benign rationale
    backend: str = "rule"  # "rule" (deterministic) or "llm" (Claude)
    defense: bool = True  # enable challenge/freeze in the graph
    threshold: float = 3.0  # deviation-score alert threshold (calibrate in practice)
    vote: bool = False  # require peer majority before freezing (N>=3)
    seed: int = 0


class IncidentState(TypedDict, total=False):
    """LangGraph state. List fields use an ``operator.add`` reducer so each node
    returns only its own appended items."""

    email: Email
    config: RunConfig
    trajectory: Annotated[list[Step], operator.add]
    deviations: Annotated[list[Deviation], operator.add]
    alerts: Annotated[list[str], operator.add]
    verdict: Optional[RiskLevel]
    chosen_action: Optional[Action]
    committed_action: Optional[Action]  # action after any containment override
    report: Optional[str]
    frozen: bool
    prevented: bool
