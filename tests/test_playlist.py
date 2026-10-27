"""Tests of the MILP of version 1 on small instances."""

import numpy as np
import pandas as pd
import pytest

from playlist import solve_v1

# Parameters with the extra constraints switched off: long tracks and new tracks are free
FREE = {"K": 10, "q_min": 0, "q_max": 1, "d_ref": 100, "s_excess": 0}


def candidates(d: list, v: list, artist: list, is_new: list | None = None) -> pd.DataFrame:
    """Candidates in the format of solve_v1."""
    is_new = is_new if is_new is not None else [False] * len(d)
    return pd.DataFrame({"d": d, "v": v, "artist": artist, "is_new": is_new})


def test_small_optimum():
    """Room for two tracks of 10 minutes: the two with the highest value."""
    cand = candidates([10] * 4, [9, 5, 8, 7], list("abcd"))
    res = solve_v1(cand, T_min=20, T_max=20, **FREE)
    assert res["status"] == "Optimal"
    assert res["x"].round().tolist() == [1, 0, 1, 0]
    assert res["obj"] == pytest.approx(17)


def test_artist_limit():
    """The two best tracks share an artist: with K = 1 only one of them."""
    cand = candidates([10] * 4, [9, 8, 5, 4], list("aabc"))
    res = solve_v1(cand, T_min=20, T_max=20, **dict(FREE, K=1))
    assert res["x"].round().tolist() == [1, 0, 1, 0]


def test_share_of_new_tracks():
    """The new tracks are the worst: with q_min = 0.5 half of the playlist is new anyway."""
    cand = candidates([10] * 4, [9, 8, 2, 1], list("abcd"), is_new=[False, False, True, True])
    res = solve_v1(cand, T_min=20, T_max=20, **dict(FREE, q_min=0.5))
    assert res["x"].round().tolist() == [1, 0, 1, 0]


def test_constraints_on_a_random_instance():
    """All the constraints hold on a larger random instance."""
    rng = np.random.default_rng(1)
    n = 200
    d = rng.uniform(2, 7, n)
    artist = rng.choice(list("abcdefgh"), n)
    cand = candidates(d, d * rng.uniform(0.8, 1, n), artist, is_new=rng.random(n) < 0.2)
    params = {
        "T_min": 40, "T_max": 45, "K": 2, "q_min": 0.1, "q_max": 0.4, "d_ref": 4, "s_excess": 0.05,
    }
    res = solve_v1(cand, **params)
    assert res["status"] == "Optimal"
    chosen = cand[res["x"] > 0.5]
    total = chosen["d"].sum()
    assert 40 - 1e-6 <= total <= 45 + 1e-6
    assert chosen["artist"].value_counts().max() <= 2
    assert 0.1 <= chosen["is_new"].mean() <= 0.4
    assert np.maximum(chosen["d"] - 4, 0).sum() <= 0.05 * total + 1e-6

    # The LP relaxation is an upper bound of the MILP
    relaxed = solve_v1(cand, **params, relax=True)
    assert relaxed["obj"] >= res["obj"] - 1e-6


def test_infeasible():
    """50 minutes requested, 20 available."""
    cand = candidates([10, 10], [9, 8], list("ab"))
    res = solve_v1(cand, T_min=50, T_max=60, **FREE)
    assert res["status"] == "Infeasible"
