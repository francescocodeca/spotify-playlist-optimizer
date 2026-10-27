"""Tests of the analysis of the experiment on small hand-made cases."""

import numpy as np
import pandas as pd
import pytest

from analysis import bootstrap_ci, effect, experiment_plays, permutation_test, rate_difference

T0 = pd.Timestamp("2026-11-02 06:00", tz="UTC")


def test_experiment_plays():
    """Experimental plays: track of the active playlist, in a run of 3 in the same session."""
    week_1, week_2 = T0, T0 + pd.Timedelta(days=7)
    playlists = pd.DataFrame({
        "uploaded": [week_1] * 4 + [week_2] * 3,
        "track_uri": ["a", "b", "c", "d", "e", "f", "g"],
    })
    plays = [
        # Before the first upload: never experimental
        (week_1 - pd.Timedelta(hours=1), "a", 0),
        # Run of 3 in the playlist, then a track outside it and a single playlist track
        (week_1 + pd.Timedelta(hours=1), "a", 1), (week_1 + pd.Timedelta(hours=1.1), "b", 1),
        (week_1 + pd.Timedelta(hours=1.2), "c", 1), (week_1 + pd.Timedelta(hours=1.3), "x", 1),
        (week_1 + pd.Timedelta(hours=1.4), "d", 1),
        # Run of 3 split by the end of a session: 2 + 1
        (week_1 + pd.Timedelta(hours=5), "a", 2), (week_1 + pd.Timedelta(hours=5.1), "b", 2),
        (week_1 + pd.Timedelta(hours=9), "c", 3),
        # Week 2: a, b, c belong to the old playlist; e, f, g form a run
        (week_2 + pd.Timedelta(hours=1), "a", 4), (week_2 + pd.Timedelta(hours=1.1), "b", 4),
        (week_2 + pd.Timedelta(hours=1.2), "c", 4), (week_2 + pd.Timedelta(hours=1.3), "e", 4),
        (week_2 + pd.Timedelta(hours=1.4), "f", 4), (week_2 + pd.Timedelta(hours=1.5), "g", 4),
    ]
    df = pd.DataFrame(plays, columns=["ts", "track_uri", "session_id"])
    flags = experiment_plays(df, playlists, min_run=3)
    expected = [False] + [True] * 3 + [False] * 2 + [False] * 3 + [False] * 3 + [True] * 3
    assert flags.tolist() == expected


def test_rate_difference():
    """Control rate minus MILP rate, for one labelling or one per row."""
    y = np.array([1, 0, 0, 0, 1, 1, 0, 0])
    milp = np.array([True, True, True, True, False, False, False, False])
    # Control 2/4 minus MILP 1/4
    assert rate_difference(y, milp) == pytest.approx(0.25)
    # One labelling per row of a matrix
    both = np.vstack([milp, ~milp])
    assert rate_difference(y, both) == pytest.approx([0.25, -0.25])


def test_effect():
    """Difference of the rates and odds ratio."""
    out = effect(0.10, 0.05)
    assert out["difference"] == pytest.approx(0.05)
    assert out["odds_ratio"] == pytest.approx((0.05 / 0.95) / (0.10 / 0.90))


def simulated(odds_ratio: float, n_sessions: int = 100, seed: int = 0) -> tuple:
    """Plays of n_sessions sessions of 10 tracks, arm at random, skip rate 0.3 in the control."""
    rng = np.random.default_rng(seed)
    session = np.repeat(np.arange(n_sessions), 10)
    milp = rng.random(len(session)) < 0.5
    odds = 0.3 / 0.7 * np.where(milp, odds_ratio, 1)
    y = (rng.random(len(session)) < odds / (1 + odds)).astype(float)
    return y, milp, session


def test_permutation_test_finds_a_strong_effect():
    """Odds ratio 0.3 on 1,000 plays: p-value small."""
    y, milp, session = simulated(odds_ratio=0.3)
    observed, p = permutation_test(y, milp, session, n_perm=2_000, rng=0)
    assert observed > 0
    assert p < 0.01


def test_permutation_test_without_effect():
    """No effect: p-value not small."""
    y, milp, session = simulated(odds_ratio=1)
    _, p = permutation_test(y, milp, session, n_perm=2_000, rng=0)
    assert p > 0.05


def test_permutation_test_stays_within_sessions():
    """Each session has only one arm: permuting within sessions changes nothing, p = 1."""
    y, _, session = simulated(odds_ratio=1)
    milp = session % 2 == 0
    y = np.where(milp, 0.0, y)
    _, p = permutation_test(y, milp, session, n_perm=500, rng=0)
    assert p == 1


def test_permutation_test_is_reproducible():
    """Same seed, same result."""
    y, milp, session = simulated(odds_ratio=1)
    assert permutation_test(y, milp, session, n_perm=500, rng=7) == permutation_test(
        y, milp, session, n_perm=500, rng=7
    )


def test_bootstrap_ci_contains_the_estimate():
    """Each interval contains the point estimate."""
    y, milp, session = simulated(odds_ratio=0.5)
    estimate = effect(y[~milp].mean(), y[milp].mean())
    ci = bootstrap_ci(y, milp, session, n_boot=1_000, rng=0)
    for name, value in estimate.items():
        assert ci.loc[name, "ci_low"] <= value <= ci.loc[name, "ci_high"]
