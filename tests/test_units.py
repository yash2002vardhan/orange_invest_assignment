"""Contract tests for the mock API and unit tests for small components."""

from __future__ import annotations

from collections import Counter

import pytest

from errand_agent.budget import Budget
from errand_agent.clock import FakeClock
from errand_agent.guard import HeuristicGuard
from errand_agent.llm import route
from errand_agent.mock_api import INJECTION_REVIEWS, MockProviderAPI, Scenario, ToolError
from errand_agent.tracing import sanitize


@pytest.fixture
def api(tmp_path):
    return MockProviderAPI(tmp_path / "p.db", FakeClock(), Scenario(seed=7))


def test_search_returns_3_to_8_providers(api):
    for service in ("plumber", "electrician", "ac_servicing", "cleaning", "carpenter"):
        for area in ("indiranagar", "koramangala", "hsr_layout", "whitefield"):
            n = len(api.search_providers(service, area, ("2026-11-03", "2026-11-05")))
            assert 3 <= n <= 8


def test_some_review_contains_injection(api):
    reviews = [r for pid in ("PLU-IND-1", "ELE-KOR-1") for r in api.get_provider(pid)["reviews"]]
    assert any(r in INJECTION_REVIEWS for r in reviews)


def test_book_outcome_distribution_is_seeded(tmp_path):
    def draws(seed):
        a = MockProviderAPI(tmp_path / f"d{seed}.db", FakeClock(), Scenario(seed=seed))
        return Counter(a._next_book_outcome() for _ in range(4000))
    c = draws(1)
    assert c == draws(1)  # reproducible
    assert 0.66 < c["success"] / 4000 < 0.74
    assert 0.12 < c["slot_taken"] / 4000 < 0.18
    assert 0.12 < (c["timeout_booked"] + c["timeout_not_booked"]) / 4000 < 0.18


def test_hold_expires_after_10_minutes(tmp_path):
    clock = FakeClock()
    a = MockProviderAPI(tmp_path / "p.db", clock, Scenario(all_slots_free=True, book_outcomes=["success"]))
    hold = a.hold_slot("PLU-IND-1", "2026-11-03T10:00")
    clock.advance(601)
    with pytest.raises(ToolError) as e:
        a.book(hold["hold_id"], "k1")
    assert e.value.code == "hold_expired"


def test_book_is_idempotent_on_key(tmp_path):
    a = MockProviderAPI(tmp_path / "p.db", FakeClock(), Scenario(all_slots_free=True, book_outcomes=["success"] * 3))
    hold = a.hold_slot("PLU-IND-1", "2026-11-03T10:00")
    assert a.book(hold["hold_id"], "k1") == a.book(hold["hold_id"], "k1")
    assert a.get_booking_status("k1")["status"] == "booked"
    assert a.get_booking_status("nope") == {"status": "not_found"}


def test_held_slot_is_not_free_for_others(tmp_path):
    a = MockProviderAPI(tmp_path / "p.db", FakeClock(), Scenario(all_slots_free=True))
    a.hold_slot("PLU-IND-1", "2026-11-03T10:00")
    with pytest.raises(ToolError) as e:
        a.hold_slot("PLU-IND-1", "2026-11-03T10:00")
    assert e.value.code == "slot_unavailable"


def test_router():
    assert route("2", "choosing")[0] == "cheap"
    assert route("yes please", "collecting")[0] == "cheap"
    assert route("actually move it to Saturday", "idle")[0] == "strong"
    assert route("Book the same electrician as last time, but not before 10 a.m.", "idle")[0] == "strong"


def test_heuristic_guard_flags_all_injections():
    verdicts = HeuristicGuard().check(INJECTION_REVIEWS + ["Arrived on time and fixed the issue quickly."])
    assert [v.flagged for v in verdicts] == [True, True, True, False]


def test_sanitize_redacts_and_truncates():
    out = sanitize({"api_key": "x", "msg": "token sk-abcdefgh12345 here", "long": "a" * 1000, "xs": list(range(20))},
                   _secrets=[])
    assert out["api_key"] == "[REDACTED]" and "sk-" not in out["msg"]
    assert len(out["long"]) < 300 and len(out["xs"]) == 9


def test_invariant_checker_catches_unapproved_booking(tmp_path):
    from harness import Harness

    h = Harness(tmp_path, scenario=Scenario(all_slots_free=True, book_outcomes=["success"]))
    h.agent.gateway.budget = Budget(h.limits, h.clock, h.tracer)
    hold = h.agent.gateway.hold_slot("PLU-IND-1", "2026-11-03T10:00")
    # bypass the approval gate entirely, as a buggy agent might
    h.agent.gateway._call("book", h.api.book, hold_id=hold["hold_id"], idempotency_key="rogue")
    with pytest.raises(AssertionError, match="without an approval check"):
        h.check_invariants()


def test_grant_is_compare_and_swap(tmp_path):
    from errand_agent.store import Store

    s = Store(tmp_path / "a.db")
    a = s.create_approval("u1", "book", "PLU-IND-1", "2026-11-03T10:00", 600)
    assert s.grant_approval(a.id) is True
    assert s.grant_approval(a.id) is False  # a second concurrent "yes" cannot grant again
