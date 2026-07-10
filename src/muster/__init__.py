"""Muster: detecting hijacked teammates in a multi-agent SOC.

A prompt-injection-hijacked agent takes actions that are locally plausible but
betray the team in context. Muster gives each agent an in-context model of its
peers' expected policy and flags departures from it -- the dual of ad-hoc
teamwork, applied to agentic AI security.
"""

from .state import (
    ACTIONS,
    AGENT_ROLES,
    RISK_LEVELS,
    Deviation,
    Email,
    IncidentState,
    RunConfig,
    Step,
)

__version__ = "0.1.0"

__all__ = [
    "ACTIONS",
    "AGENT_ROLES",
    "RISK_LEVELS",
    "Deviation",
    "Email",
    "IncidentState",
    "RunConfig",
    "Step",
    "__version__",
]
