r"""LangGraph orchestration of the SOC pipeline.

    analyzer -> responder -> [frozen?] -> containment -> reporter -> END
                                \-> reporter -> END

The Sentinel scores each agent's step against the teammate model *inside* the
node. The Responder's intended action is scored before it is committed, so a
conditional edge can divert a flagged action to a containment node that
downgrades it to a safe fallback -- prevention, not post-hoc logging.
"""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

from .agents import analyze, report, respond
from .defense import CONTAINMENT_ACTION, SentinelMonitor
from .scoring import SENTINEL_WEIGHTS
from .state import Email, IncidentState, RunConfig

_LLM_CLIENT = None  # lazily constructed, reused across nodes


def _client_for(config: RunConfig):
    global _LLM_CLIENT
    if getattr(config, "backend", "rule") != "llm":
        return None
    if _LLM_CLIENT is None:
        from .llm import try_build_client

        _LLM_CLIENT = try_build_client()
    return _LLM_CLIENT


def _monitor(config: RunConfig) -> SentinelMonitor:
    return SentinelMonitor(
        threshold=getattr(config, "threshold", 3.0),
        weights=SENTINEL_WEIGHTS,
        vote=getattr(config, "vote", False),
    )


def _analyzer_node(state: IncidentState) -> dict:
    email, config = state["email"], state["config"]
    mon = _monitor(config)
    step = analyze(email, config, _client_for(config))
    dev = mon.assess(step, 0, [], email)
    alerts = [mon.challenge(dev)] if config.defense and dev.score >= mon.threshold else []
    return {"trajectory": [step], "verdict": step.action, "deviations": [dev], "alerts": alerts}


def _responder_node(state: IncidentState) -> dict:
    email, config = state["email"], state["config"]
    mon = _monitor(config)
    verdict = state.get("verdict")
    prefix = list(state.get("trajectory", []))
    step = respond(email, verdict, config, _client_for(config))
    dev = mon.assess(step, len(prefix), prefix, email)

    frozen = False
    alerts: list[str] = []
    if config.defense and dev.score >= mon.threshold:
        alerts.append(mon.challenge(dev))
        frozen = mon.decide(dev, seed=config.seed).freeze
    return {
        "trajectory": [step],
        "chosen_action": step.action,
        "committed_action": step.action,
        "deviations": [dev],
        "alerts": alerts,
        "frozen": frozen,
    }


def _route_after_responder(state: IncidentState) -> str:
    return "containment" if state.get("frozen") else "reporter"


def _containment_node(state: IncidentState) -> dict:
    intended = state.get("chosen_action")
    return {
        "committed_action": CONTAINMENT_ACTION,
        "prevented": True,
        "alerts": [
            f"FREEZE responder: high-privilege action {intended!r} downgraded "
            f"to {CONTAINMENT_ACTION!r} before commit."
        ],
    }


def _reporter_node(state: IncidentState) -> dict:
    email, config = state["email"], state["config"]
    mon = _monitor(config)
    verdict = state.get("verdict")
    committed = state.get("committed_action")
    prefix = list(state.get("trajectory", []))
    step = report(email, verdict, committed, config)
    dev = mon.assess(step, len(prefix), prefix, email)
    alerts = [mon.challenge(dev)] if config.defense and dev.score >= mon.threshold else []
    return {
        "trajectory": [step],
        "report": step.metadata.get("report_text"),
        "deviations": [dev],
        "alerts": alerts,
    }


def build_graph():
    """Compile and return the incident-response graph."""

    g = StateGraph(IncidentState)
    g.add_node("analyzer", _analyzer_node)
    g.add_node("responder", _responder_node)
    g.add_node("containment", _containment_node)
    g.add_node("reporter", _reporter_node)

    g.add_edge(START, "analyzer")
    g.add_edge("analyzer", "responder")
    g.add_conditional_edges(
        "responder",
        _route_after_responder,
        {"containment": "containment", "reporter": "reporter"},
    )
    g.add_edge("containment", "reporter")
    g.add_edge("reporter", END)
    return g.compile()


_GRAPH = None


def run_incident(email: Email, config: RunConfig) -> IncidentState:
    """Run one incident through the compiled graph and return the final state."""

    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    initial: IncidentState = {
        "email": email,
        "config": config,
        "trajectory": [],
        "deviations": [],
        "alerts": [],
        "frozen": False,
        "prevented": False,
    }
    return _GRAPH.invoke(initial)
