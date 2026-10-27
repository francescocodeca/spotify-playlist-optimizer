"""Optimized playlists with a MILP (PuLP 4 + HiGHS solver).

Version 1: choice of the tracks of one playlist.
Rotation across days: every day version 1, without the tracks used in the previous
K_DAYS - 1 days.
From the terminal: python src/playlist.py "2026-10-26 08:00" --minutes 45 [--save]
"""

import argparse

import joblib
import numpy as np
import pandas as pd
import pulp

from features import ROOT
from train import MODEL_PATH, candidate_features, predict_candidates

F = 0.2  # mean fraction of a track I listen to when I skip it (notebook 03)
PARAMS = {
    "T_min": 40, "T_max": 45, "K": 2, "q_min": 0.1, "q_max": 0.4, "d_ref": 4, "s_excess": 0.05,
}
# Penalty in the objective for every minute beyond d_ref: among tracks of equal value it prefers
# tracks of normal length (the s_excess constraint limits the excess, but without the penalty
# the solver uses it to fill the exact duration)
LAMBDA_EXCESS = 0.05
K_DAYS = 14  # rotation: a track appears at most once every 14 days
# Saved playlists, one per day (in data/, ignored by git)
PLAYLIST_DIR = ROOT / "data" / "playlists"


def add_day_constraints(
    prob: pulp.LpProblem, x: list, cand: pd.DataFrame, T_min: float, T_max: float, K: int,
    q_min: float, q_max: float, d_ref: float, s_excess: float, tag: str = "",
) -> None:
    """Constraints of a single playlist. x: one variable per row of cand.

    tag: suffix of the constraint names (needed when the same problem holds several days).
    """
    I = range(len(cand))
    d = cand["d"].to_numpy()
    new = cand["is_new"].to_numpy()

    total = pulp.lpSum(d[i] * x[i] for i in I)
    prob += total >= T_min, f"duration_min{tag}"
    prob += total <= T_max, f"duration_max{tag}"

    # At most K tracks per artist (only artists with more than K candidates can violate it)
    groups = [idx for idx in cand.groupby("artist").indices.values() if len(idx) > K]
    for j, idx in enumerate(groups):
        prob += pulp.lpSum(x[i] for i in idx) <= K, f"artist_{j}{tag}"

    # Minutes beyond d_ref: at most a share s_excess of the playlist
    excess = np.maximum(d - d_ref, 0)
    prob += pulp.lpSum(excess[i] * x[i] for i in I) <= s_excess * total, f"excess{tag}"

    # Share of new tracks between q_min and q_max (ratio linearized: multiplied by the count)
    n_sel = pulp.lpSum(x[i] for i in I)
    n_new = pulp.lpSum(x[i] for i in I if new[i])
    prob += n_new >= q_min * n_sel, f"new_min{tag}"
    prob += n_new <= q_max * n_sel, f"new_max{tag}"


def solve_v1(
    cand: pd.DataFrame, T_min: float, T_max: float, K: int, q_min: float, q_max: float,
    d_ref: float, s_excess: float, relax: bool = False, lam_excess: float = LAMBDA_EXCESS,
) -> dict:
    """MILP version 1. cand: columns d (minutes), v (expected minutes), is_new, artist.

    relax=True: LP relaxation. lam_excess: penalty per minute beyond d_ref (0 = none).
    """
    I = range(len(cand))
    v = cand["v"].to_numpy()
    excess = np.maximum(cand["d"].to_numpy() - d_ref, 0)
    prob = pulp.LpProblem("playlist_v1", pulp.LpMaximize)
    cat = "Continuous" if relax else "Binary"
    x = [prob.add_variable(f"x_{i}", lowBound=0, upBound=1, cat=cat) for i in I]
    prob += pulp.lpSum((v[i] - lam_excess * excess[i]) * x[i] for i in I)
    add_day_constraints(prob, x, cand, T_min, T_max, K, q_min, q_max, d_ref, s_excess)
    stats = prob.solve(pulp.HiGHS(msg=False))
    xs = np.array([x[i].value() for i in I])
    return {
        "status": stats.status_str, "obj": stats.objective, "x": xs, "secs": stats.time,
        "n_constraints": prob.numConstraints(),
    }


def recent_tracks(streams: pd.DataFrame, when: pd.Timestamp, k_days: int = K_DAYS) -> set:
    """Tracks to exclude on the day of `when`: played (history) or put in a saved playlist
    in the previous k_days - 1 days. With k_days = 14 a Monday track returns two Mondays later.
    """
    today = when.normalize()  # midnight of the playlist day
    first = today - pd.Timedelta(days=k_days - 1)  # first excluded day
    recent = set(streams.loc[streams["ts"] >= first, "track_uri"])
    for path in PLAYLIST_DIR.glob("*.csv"):
        day = pd.Timestamp(path.stem, tz="Europe/Berlin")  # file name = date of the playlist
        if first <= day < today:
            recent |= set(pd.read_csv(path)["track_uri"])
    return recent


def predict_pool(
    streams: pd.DataFrame, pool: pd.DataFrame, model, when: pd.Timestamp, **context,
) -> pd.DataFrame:
    """Predictions for the candidates in `pool`, with duration d and expected minutes v.

    context: platform, country, shuffle, offline (see train.candidate_features).
    """
    feats = candidate_features(streams, pool["track_uri"], when, **context)
    cand = predict_candidates(model, feats)
    cand = cand.merge(pool[["track_uri", "duration_s", "is_new"]], on="track_uri")
    cand["d"] = cand["duration_s"] / 60
    cand["v"] = (1 - (1 - F) * cand["p0"]) * cand["d"]
    return cand


def daily_playlist(
    streams: pd.DataFrame, pool: pd.DataFrame, model, when, minutes: int = 45,
    k_days: int = K_DAYS, **context,
) -> pd.DataFrame:
    """Optimized playlist (version 1) for the moment `when` (German time if naive).

    context: platform, country, shuffle, offline (see train.candidate_features).
    """
    when = pd.Timestamp(when)
    if when.tz is None:
        when = when.tz_localize("Europe/Berlin")
    pool = pool[~pool["track_uri"].isin(recent_tracks(streams, when, k_days))]
    cand = predict_pool(streams, pool, model, when, **context)

    res = solve_v1(cand, **dict(PARAMS, T_min=minutes - 5, T_max=minutes))
    assert res["status"] == "Optimal", res["status"]
    return cand[res["x"] > 0.5].sort_values("p0")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimized playlist for a given moment.")
    parser.add_argument("when", help='German date and time, e.g. "2026-10-26 08:00"')
    parser.add_argument("--minutes", type=int, default=45, help="maximum duration in minutes")
    parser.add_argument(
        "--save", action="store_true", help="save the playlist (counts for the rotation)"
    )
    args = parser.parse_args()

    streams = pd.read_parquet(ROOT / "data" / "streams_clean.parquet")
    pool = pd.read_parquet(ROOT / "data" / "pool.parquet")
    model = joblib.load(MODEL_PATH)

    pl = daily_playlist(streams, pool, model, args.when, args.minutes)
    print(pl[["track", "artist", "d", "p0"]].round(3).to_string(index=False))
    print(f"\n{len(pl)} tracks, {pl['d'].sum():.1f} minutes, mean p0 {pl['p0'].mean():.3f}")

    if args.save:
        PLAYLIST_DIR.mkdir(exist_ok=True)
        path = PLAYLIST_DIR / f"{pd.Timestamp(args.when):%Y-%m-%d}.csv"
        pl[["track_uri", "track", "artist", "d", "p0"]].to_csv(path, index=False)
        print(f"Saved in {path.relative_to(ROOT)}")
