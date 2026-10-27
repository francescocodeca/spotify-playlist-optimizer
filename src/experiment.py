"""Weekly playlist of the A/B test (interleaving): optimized and control tracks mixed at random.

Both arms come from version 1 of the MILP with the same constraints; the control arm has a random
objective, so the two arms differ only in the use of the predictions.
From the terminal: python src/experiment.py [2026-11-02] [--upload] (default: Monday of this week)
"""

import argparse
import sys

import joblib
import numpy as np
import pandas as pd

from features import ROOT
from playlist import PARAMS, predict_pool, solve_v1
from spotify_client import experiment_playlist, get_client
from train import MODEL_PATH

ARM_MINUTES = 300  # minutes per arm in each weekly playlist
EXCLUDE_WEEKS = 4  # a track returns at most once every 4 weeks (both arms)
# Weekly playlists with the arm of every track (in data/, ignored by git): not to be opened
# during the experiment, so that the test stays blind
EXPERIMENT_DIR = ROOT / "data" / "experiment"
API_BATCH = 100  # maximum number of tracks per request to the Spotify API


def past_tracks(monday: pd.Timestamp, weeks: int = EXCLUDE_WEEKS) -> set:
    """Tracks of the experiment playlists of the `weeks` weeks before `monday`."""
    first = monday - pd.Timedelta(weeks=weeks)
    used = set()
    for path in EXPERIMENT_DIR.glob("*.csv"):
        if first <= pd.Timestamp(path.stem) < monday:
            used |= set(pd.read_csv(path)["track_uri"])
    return used


def weekly_playlist(
    streams: pd.DataFrame, pool: pd.DataFrame, model, monday: pd.Timestamp,
    minutes: int = ARM_MINUTES, exclude: set | None = None,
) -> pd.DataFrame:
    """Both arms of the week, in random order. Columns: track_uri, arm, d, p0, is_new.

    exclude: tracks not allowed this week; by default the tracks of the experiment playlists of
    the previous EXCLUDE_WEEKS weeks (files in EXPERIMENT_DIR).
    """
    # Seed from the date: the same week always gives the same playlist (reproducible)
    rng = np.random.default_rng(int(f"{monday:%Y%m%d}"))
    if exclude is None:
        exclude = past_tracks(monday)
    pool = pool[~pool["track_uri"].isin(exclude)]
    when = (monday + pd.Timedelta(hours=8)).tz_localize("Europe/Berlin")
    cand = predict_pool(streams, pool, model, when).reset_index(drop=True)
    params = dict(PARAMS, T_min=minutes - 5, T_max=minutes)

    # 1. Optimized arm: version 1 of the MILP
    res = solve_v1(cand, **params)
    assert res["status"] == "Optimal", res["status"]
    milp = cand[res["x"] > 0.5].assign(arm="milp")

    # 2. Control arm: same MILP on the remaining tracks, with a random value per minute
    rest = cand[res["x"] < 0.5].reset_index(drop=True)
    rest["v"] = rng.random(len(rest)) * rest["d"]
    res = solve_v1(rest, **params)
    assert res["status"] == "Optimal", res["status"]
    control = rest[res["x"] > 0.5].assign(arm="control")

    # 3. Random order: the position in the playlist does not reveal the arm
    both = pd.concat([milp, control]).sample(frac=1, random_state=rng)
    return both[["track_uri", "arm", "d", "p0", "is_new"]].reset_index(drop=True)


def plan_weeks(
    streams: pd.DataFrame, pool: pd.DataFrame, model, mondays: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Weekly playlists of several Mondays built in memory, with the same 4-week rotation.

    The files in EXPERIMENT_DIR are neither read nor written. Columns of weekly_playlist plus
    monday.
    """
    weeks = {}
    for monday in mondays:
        first = monday - pd.Timedelta(weeks=EXCLUDE_WEEKS)
        exclude = set().union(*(set(w["track_uri"]) for m, w in weeks.items() if m >= first))
        weeks[monday] = weekly_playlist(streams, pool, model, monday, exclude=exclude)
    return pd.concat(weeks, names=["monday", None]).reset_index(level=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Weekly playlist of the experiment.")
    parser.add_argument(
        "monday", nargs="?", help="Monday of the week, e.g. 2026-11-02 (default: this week)"
    )
    parser.add_argument(
        "--upload", action="store_true", help="save the playlist and send it to Spotify"
    )
    args = parser.parse_args()
    # Default: Monday of the current week in German time (for the automatic weekly run)
    if args.monday:
        monday = pd.Timestamp(args.monday)
    else:
        today = pd.Timestamp.now(tz="Europe/Berlin").tz_localize(None).normalize()
        monday = today - pd.Timedelta(days=today.day_of_week)
    assert monday.day_of_week == 0, "the date must be a Monday"
    print(f"{pd.Timestamp.now(tz='Europe/Berlin'):%Y-%m-%d %H:%M}, week of {monday:%Y-%m-%d}")
    path = EXPERIMENT_DIR / f"{monday:%Y-%m-%d}.csv"
    # An upload is never repeated: a second upload time would hide the plays between the two
    if args.upload and path.exists():
        print(f"Already uploaded ({path.relative_to(ROOT)}): nothing to do")
        sys.exit()

    streams = pd.read_parquet(ROOT / "data" / "streams_clean.parquet")
    pool = pd.read_parquet(ROOT / "data" / "pool.parquet")
    model = joblib.load(MODEL_PATH)
    pl = weekly_playlist(streams, pool, model, monday)

    # Only totals per arm: which track is in which arm stays hidden (blind test)
    summary = pl.groupby("arm").agg(
        tracks=("track_uri", "size"), minutes=("d", "sum"), mean_p0=("p0", "mean"),
        new_share=("is_new", "mean"),
    )
    print(summary.round(3).to_string())

    if args.upload:
        EXPERIMENT_DIR.mkdir(exist_ok=True)
        pl.assign(uploaded=pd.Timestamp.now(tz="Europe/Berlin")).to_csv(path, index=False)

        sp = get_client()
        playlist_id = experiment_playlist(sp)
        uris = pl["track_uri"].tolist()
        # The API takes at most 100 tracks per request: replace with the first 100, then add
        sp.playlist_replace_items(playlist_id, uris[:API_BATCH])
        for i in range(API_BATCH, len(uris), API_BATCH):
            sp.playlist_add_items(playlist_id, uris[i:i + API_BATCH])
        print(f"Saved in {path.relative_to(ROOT)}, {len(uris)} tracks sent to Spotify")
