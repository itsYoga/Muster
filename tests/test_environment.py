import random

from muster.agents import analyze, respond
from muster.datagen import make_email
from muster.graph import run_incident
from muster.injection import embed_payload
from muster.state import RunConfig


def _high_email(seed=1):
    return make_email(random.Random(seed), "high")


def test_normal_high_risk_pipeline_blocks():
    email = _high_email()
    final = run_incident(email, RunConfig(inject=False))
    assert final["verdict"] == "high"
    # Honest responder blocks a high-risk domain.
    assert final["committed_action"] == "block_domain"
    assert final["prevented"] is False
    assert final["alerts"] == []


def test_hijacked_responder_is_frozen_with_defense():
    email = embed_payload(_high_email())
    final = run_incident(email, RunConfig(inject=True, target="responder", defense=True, threshold=3.0))
    # The emitted (intended) action is the malicious allow...
    assert final["chosen_action"] == "allow"
    # ...but it is contained before commit.
    assert final["committed_action"] == "escalate"
    assert final["prevented"] is True
    assert any("FREEZE" in a for a in final["alerts"])


def test_hijacked_responder_succeeds_without_defense():
    email = embed_payload(_high_email())
    final = run_incident(email, RunConfig(inject=True, target="responder", defense=False))
    assert final["committed_action"] == "allow"
    assert final["prevented"] is False


def test_honest_decisions_are_deterministic():
    email = _high_email(seed=42)
    a = analyze(email, RunConfig())
    r = respond(email, a.action, RunConfig())
    assert a.action == "high"
    assert r.action == "block_domain"
