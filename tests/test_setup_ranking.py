"""
Tests for SetupRanker (Phase 13).

The ranker scores detected DayTradeSetup candidates and orders them best-first.
It never places orders or touches the broker — it only assigns rank/score
metadata on copies and returns a sorted list.

Coverage:
  - score formula and descending sort
  - higher confidence ranks higher
  - invalid setups rank last (penalty)
  - filter_top_setups removes weak / invalid setups and caps results
  - rank numbers assigned 1..N
  - the original input list and objects are never mutated
"""
from __future__ import annotations

import copy

import pytest

from strategy.day_trade_setups import DayTradeSetup, SetupType
from strategy.setup_ranking import SetupRanker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_setup(
    ticker: str = "SPY",
    setup_type: SetupType = SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG,
    is_valid: bool = True,
    confidence: float = 0.7,
    risk_reward_1: float = 2.0,
    risk_reward_2: float = 3.0,
    unusual_volume: float = 0.0,
) -> DayTradeSetup:
    """Build a DayTradeSetup with the fields that matter to ranking."""
    return DayTradeSetup(
        ticker=ticker,
        timestamp="2026-06-05T10:00:00",
        setup_type=setup_type,
        is_valid=is_valid,
        confidence=confidence,
        entry_trigger=100.0,
        stop_loss=99.0,
        target_1=102.0,
        target_2=103.0,
        risk_reward_1=risk_reward_1,
        risk_reward_2=risk_reward_2,
        reason="test setup",
        warnings=[],
        metadata={"unusual_volume": unusual_volume},
    )


@pytest.fixture
def ranker() -> SetupRanker:
    return SetupRanker()


# ---------------------------------------------------------------------------
# Score formula
# ---------------------------------------------------------------------------

def test_score_formula_basic(ranker):
    """Score = confidence*2 + rr1*0.5 + rr2*0.25 for a plain valid setup."""
    setup = _make_setup(
        setup_type=SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG,
        confidence=0.7, risk_reward_1=2.0, risk_reward_2=3.0, unusual_volume=0.0,
    )
    # 0.7*2 + 2*0.5 + 3*0.25 = 1.4 + 1.0 + 0.75 = 3.15
    assert ranker.score_setup(setup) == pytest.approx(3.15)


def test_score_orb_bonus(ranker):
    """OPENING_RANGE_BREAKOUT_LONG adds +0.5."""
    base = _make_setup(setup_type=SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG)
    orb = _make_setup(setup_type=SetupType.OPENING_RANGE_BREAKOUT_LONG)
    assert ranker.score_setup(orb) == pytest.approx(ranker.score_setup(base) + 0.5)


def test_score_vwap_bonus(ranker):
    """VWAP_RECLAIM_LONG adds +0.25."""
    base = _make_setup(setup_type=SetupType.HIGH_RELATIVE_VOLUME_CONTINUATION_LONG)
    vwap = _make_setup(setup_type=SetupType.VWAP_RECLAIM_LONG)
    assert ranker.score_setup(vwap) == pytest.approx(ranker.score_setup(base) + 0.25)


def test_score_unusual_volume_bonus(ranker):
    """unusual_volume == 1.0 adds +0.5."""
    base = _make_setup(unusual_volume=0.0)
    unusual = _make_setup(unusual_volume=1.0)
    assert ranker.score_setup(unusual) == pytest.approx(ranker.score_setup(base) + 0.5)


def test_score_invalid_penalty(ranker):
    """is_valid == False subtracts 1.0."""
    valid = _make_setup(is_valid=True)
    invalid = _make_setup(is_valid=False)
    assert ranker.score_setup(invalid) == pytest.approx(ranker.score_setup(valid) - 1.0)


# ---------------------------------------------------------------------------
# Ranking / sorting
# ---------------------------------------------------------------------------

def test_ranking_sorts_descending(ranker):
    """rank_setups returns setups ordered from highest score to lowest."""
    low = _make_setup(ticker="LOW", confidence=0.3, risk_reward_1=1.0, risk_reward_2=1.0)
    mid = _make_setup(ticker="MID", confidence=0.6, risk_reward_1=2.0, risk_reward_2=2.5)
    high = _make_setup(ticker="HIGH", confidence=0.9, risk_reward_1=3.0, risk_reward_2=4.0)

    ranked = ranker.rank_setups([low, high, mid])

    tickers = [s.ticker for s in ranked]
    assert tickers == ["HIGH", "MID", "LOW"]

    scores = [s.metadata["score"] for s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_higher_confidence_ranks_higher(ranker):
    """With everything else equal, higher confidence ranks first."""
    low_conf = _make_setup(ticker="LOWC", confidence=0.4)
    high_conf = _make_setup(ticker="HIGHC", confidence=0.95)

    ranked = ranker.rank_setups([low_conf, high_conf])

    assert ranked[0].ticker == "HIGHC"
    assert ranked[0].metadata["rank"] == 1
    assert ranked[1].metadata["rank"] == 2


def test_invalid_setups_rank_last(ranker):
    """Invalid setups (with the -1 penalty) sort below valid ones."""
    invalid = _make_setup(ticker="BAD", is_valid=False, confidence=0.9)
    valid = _make_setup(ticker="GOOD", is_valid=True, confidence=0.6)

    ranked = ranker.rank_setups([invalid, valid])

    assert ranked[0].ticker == "GOOD"
    assert ranked[-1].ticker == "BAD"


def test_rank_numbers_assigned(ranker):
    """metadata['rank'] runs 1..N with no gaps or duplicates, 1 = best."""
    setups = [
        _make_setup(ticker="A", confidence=0.9),
        _make_setup(ticker="B", confidence=0.7),
        _make_setup(ticker="C", confidence=0.5),
    ]
    ranked = ranker.rank_setups(setups)

    ranks = [s.metadata["rank"] for s in ranked]
    assert ranks == [1, 2, 3]


def test_rank_empty_list(ranker):
    """Ranking an empty list returns an empty list."""
    assert ranker.rank_setups([]) == []


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def test_filter_removes_weak_setups(ranker):
    """filter_top_setups drops setups below min_confidence."""
    weak = _make_setup(ticker="WEAK", confidence=0.4)
    strong = _make_setup(ticker="STRONG", confidence=0.8)

    top = ranker.filter_top_setups([weak, strong], min_confidence=0.6)

    tickers = [s.ticker for s in top]
    assert "STRONG" in tickers
    assert "WEAK" not in tickers


def test_filter_removes_invalid_setups(ranker):
    """filter_top_setups drops invalid setups even with high confidence."""
    invalid = _make_setup(ticker="BAD", is_valid=False, confidence=0.9)
    valid = _make_setup(ticker="GOOD", is_valid=True, confidence=0.7)

    top = ranker.filter_top_setups([invalid, valid], min_confidence=0.6)

    assert [s.ticker for s in top] == ["GOOD"]


def test_filter_caps_max_results(ranker):
    """filter_top_setups returns at most max_results setups."""
    setups = [_make_setup(ticker=f"T{i}", confidence=0.9) for i in range(10)]

    top = ranker.filter_top_setups(setups, min_confidence=0.6, max_results=3)

    assert len(top) == 3


def test_filter_returns_best_first(ranker):
    """Filtered results stay ordered best-first."""
    setups = [
        _make_setup(ticker="MID", confidence=0.7),
        _make_setup(ticker="TOP", confidence=0.95),
        _make_setup(ticker="LOWPASS", confidence=0.61),
    ]
    top = ranker.filter_top_setups(setups, min_confidence=0.6, max_results=5)

    assert top[0].ticker == "TOP"
    assert top[0].metadata["rank"] == 1


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------

def test_does_not_mutate_input_list(ranker):
    """The original list object is not reordered or resized."""
    a = _make_setup(ticker="A", confidence=0.3)
    b = _make_setup(ticker="B", confidence=0.9)
    original = [a, b]
    snapshot = list(original)

    ranker.rank_setups(original)

    assert original == snapshot
    assert original[0] is a and original[1] is b


def test_does_not_mutate_original_objects(ranker):
    """Original setup objects get no rank/score metadata written to them."""
    setup = _make_setup(ticker="A", confidence=0.8)
    metadata_before = copy.deepcopy(setup.metadata)

    ranked = ranker.rank_setups([setup])

    # Original untouched...
    assert setup.metadata == metadata_before
    assert "rank" not in setup.metadata
    assert "score" not in setup.metadata
    # ...but the returned copy carries the new metadata.
    assert ranked[0].metadata["rank"] == 1
    assert "score" in ranked[0].metadata
    assert ranked[0] is not setup
