import random

from muster.baselines import AtomicDetector, SentinelDetector
from muster.datagen import generate_incident, make_email
from muster.injection import cover_rationale
from muster.scoring import SENTINEL_WEIGHTS, score_trajectory
from muster.state import RunConfig
from muster.text import rationale_anomaly
from muster.baselines import rebuild


def test_allow_on_high_has_high_surprisal():
    normal = generate_incident(1, inject=False)
    normal["email"]["true_risk"] = "high"  # cosmetic
    attack = generate_incident(2, inject=True, target="responder", stealth=0.0)

    email_n, traj_n = rebuild(normal)
    email_a, traj_a = rebuild(attack)
    dev_n = score_trajectory(traj_n, email_n, weights=SENTINEL_WEIGHTS)
    dev_a = score_trajectory(traj_a, email_a, weights=SENTINEL_WEIGHTS)

    responder_surprisal = dev_a[1].action_surprisal
    assert responder_surprisal > 4.0  # -log(0.01) ~ 4.6
    assert max(d.score for d in dev_a) > max(d.score for d in dev_n)


def test_stealth_suppresses_semantic_but_not_structural():
    blatant = cover_rationale("allow", stealth=0.0)
    stealthy = cover_rationale("allow", stealth=1.0)
    # Semantic anomaly collapses under stealth...
    assert rationale_anomaly(blatant) > rationale_anomaly(stealthy)
    assert rationale_anomaly(stealthy) < 0.15

    # ...but the action-surprisal is identical regardless of the wording.
    a0 = rebuild(generate_incident(3, inject=True, target="responder", stealth=0.0))
    a1 = rebuild(generate_incident(3, inject=True, target="responder", stealth=1.0))
    s0 = score_trajectory(a0[1], a0[0], weights=SENTINEL_WEIGHTS)[1].action_surprisal
    s1 = score_trajectory(a1[1], a1[0], weights=SENTINEL_WEIGHTS)[1].action_surprisal
    assert abs(s0 - s1) < 1e-9


def test_atomic_detector_misses_locally_legal_betrayal():
    attack = generate_incident(4, inject=True, target="responder", stealth=0.0)
    flagged, _ = AtomicDetector().detect(attack)
    assert flagged is False  # `allow` is a permitted Responder action


def test_sentinel_detects_after_calibration():
    normals = [generate_incident(100 + i, inject=False) for i in range(20)]
    attack = generate_incident(999, inject=True, target="responder", stealth=1.0)
    det = SentinelDetector(quantile=0.95).fit(normals)
    flagged, step = det.detect(attack)
    assert flagged is True
    assert step == 1  # caught at the responder step
