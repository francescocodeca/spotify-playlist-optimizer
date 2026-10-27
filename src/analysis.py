"""Analysis of the experiment: experimental plays, effect between the two arms and test.

From the terminal, with the new export of the streaming history (folder of the JSON files):
python src/analysis.py "<export folder>" [--count-only]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment import EXPERIMENT_DIR
from features import OBSERVED, build_features
from load_data import clean, load_raw

N_PERM = 10_000  # permutations in the test (protocol)
CHUNK = 500  # permutations computed together, to limit the memory
MIN_RUN = 3  # consecutive plays of playlist tracks needed for an experimental play (protocol)
N_BOOT = 2_000  # bootstrap resamples of the sessions for the confidence intervals
TARGET_PLAYS = 1_700  # labeled experimental plays of the stopping rule (protocol)
SEED = 2027  # seed of the random generator: the same data always give the same numbers


def load_playlists(directory: Path = EXPERIMENT_DIR) -> pd.DataFrame:
    """All weekly playlists of the experiment, one row per track and upload (times in UTC)."""
    files = sorted(directory.glob("*.csv"))
    playlists = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    playlists["uploaded"] = pd.to_datetime(playlists["uploaded"], utc=True, format="ISO8601")
    return playlists


def add_arm(df: pd.DataFrame, playlists: pd.DataFrame) -> pd.DataFrame:
    """Add uploaded, arm, d and is_new from the last upload with the track before each play.

    For an experimental play this upload is the active playlist. NaN for tracks never uploaded
    before the play. df must be sorted by time.
    """
    right = playlists[["uploaded", "track_uri", "arm", "d", "is_new"]].sort_values("uploaded")
    # For each play, the last row of `right` with the same track and uploaded <= ts
    merged = pd.merge_asof(
        df[["ts", "track_uri"]], right, left_on="ts", right_on="uploaded", by="track_uri"
    )
    merged.index = df.index
    return df.join(merged[["uploaded", "arm", "d", "is_new"]])


def experiment_plays(
    df: pd.DataFrame, playlists: pd.DataFrame, min_run: int = MIN_RUN,
) -> pd.Series:
    """Flag the experimental plays of df (True/False, same index as df).

    playlists: one row per track and upload, columns track_uri and uploaded (time of the upload).
    A play is experimental if its track is in the playlist active at its time (the last upload
    before it) and it is inside a run of at least min_run such plays in the same session.
    df must be sorted by time, with columns ts, track_uri and session_id.
    """
    uploads = pd.DatetimeIndex(playlists["uploaded"].unique()).sort_values()
    # Active playlist of each play: position of the last upload before it (-1 before the first)
    active = uploads.searchsorted(df["ts"], side="right") - 1
    pairs = pd.MultiIndex.from_arrays(
        [uploads.get_indexer(playlists["uploaded"]), playlists["track_uri"]]
    )
    in_playlist = pd.MultiIndex.from_arrays([active, df["track_uri"]]).isin(pairs)
    in_playlist = pd.Series(in_playlist & (active >= 0), index=df.index)

    # Runs: a new run starts at every play that does not continue a run of the same session
    session = df["session_id"]
    goes_on = in_playlist & in_playlist.shift(1, fill_value=False) & (session == session.shift(1))
    run = (~goes_on).cumsum()
    run_length = in_playlist.groupby(run).transform("sum")
    return in_playlist & (run_length >= min_run)


def rate_difference(y: np.ndarray, milp: np.ndarray) -> np.ndarray:
    """Skip rate of the control arm minus skip rate of the MILP arm.

    `milp` can be one vector of labels or a matrix with one labelling per row.
    """
    n_milp = milp.sum(axis=-1)
    return (y * ~milp).sum(axis=-1) / (len(y) - n_milp) - (y * milp).sum(axis=-1) / n_milp


def permutation_test(
    y: np.ndarray, milp: np.ndarray, session: np.ndarray, n_perm: int = N_PERM,
    rng: np.random.Generator | int | None = None,
) -> tuple[float, float]:
    """One-sided permutation test with the arm labels permuted within each session.

    Returns the observed difference (control minus MILP) and the p-value for H1: tracks of the
    MILP arm are skipped less often.
    """
    rng = np.random.default_rng(rng)
    # Rows sorted by session, sessions numbered 0, 1, 2, ... in the same order
    order = np.argsort(np.asarray(session), kind="stable")
    y = np.asarray(y, dtype=float)[order]
    milp = np.asarray(milp, dtype=bool)[order]
    s = pd.factorize(np.asarray(session)[order])[0]
    observed = rate_difference(y, milp)

    count = 0
    for start in range(0, n_perm, CHUNK):
        b = min(CHUNK, n_perm - start)
        # Sorting "session number + uniform noise" keeps each row in the block of its session
        # and shuffles the rows inside the block: labels permuted within each session
        idx = np.argsort(s + rng.random((b, len(s))), axis=1)
        # Small tolerance: the same labelling summed in another order can differ by rounding
        count += (rate_difference(y, milp[idx]) >= observed - 1e-12).sum()
    return observed, (1 + count) / (n_perm + 1)


def effect(rate_control: np.ndarray, rate_milp: np.ndarray) -> dict:
    """Skip rate per arm, difference (control minus MILP) and odds ratio (MILP over control)."""
    odds_control = rate_control / (1 - rate_control)
    odds_milp = rate_milp / (1 - rate_milp)
    return {
        "control": rate_control, "milp": rate_milp, "difference": rate_control - rate_milp,
        "odds_ratio": odds_milp / odds_control,
    }


def bootstrap_ci(
    y: np.ndarray, milp: np.ndarray, session: np.ndarray, n_boot: int = N_BOOT,
    rng: np.random.Generator | int | None = None,
) -> pd.DataFrame:
    """95% confidence intervals of the quantities of `effect`, resampling whole sessions.

    Whole sessions are resampled because plays of the same session are correlated (skips come in
    runs): resampling single plays would give intervals that are too narrow.
    """
    rng = np.random.default_rng(rng)
    s = pd.factorize(np.asarray(session))[0]
    y = np.asarray(y, dtype=float)
    milp = np.asarray(milp, dtype=bool)
    # Plays and skips of each arm in each session (position = session number)
    plays_milp = np.bincount(s, weights=milp)
    skips_milp = np.bincount(s, weights=y * milp)
    plays_control = np.bincount(s, weights=~milp)
    skips_control = np.bincount(s, weights=y * ~milp)

    # One resample = how many times each session is drawn (n draws with replacement):
    # one row of counts per resample, totals by matrix product
    n = len(plays_milp)
    counts = rng.multinomial(n, np.full(n, 1 / n), size=n_boot)
    rate_control = counts @ skips_control / (counts @ plays_control)
    rate_milp = counts @ skips_milp / (counts @ plays_milp)
    boot = effect(rate_control, rate_milp)
    return pd.DataFrame(
        {k: np.quantile(v, [0.025, 0.975]) for k, v in boot.items()}, index=["ci_low", "ci_high"]
    ).T


def secondary(plays: pd.DataFrame) -> pd.DataFrame:
    """Descriptive results per arm (protocol, no test), on the experimental plays.

    Fraction listened and endplay share on the plays that ended normally (OBSERVED); skip rates on
    the labeled plays, after a non-skipped track and for known and new tracks separately.
    """
    obs = plays[plays["reason_end"].isin(OBSERVED)]
    is_new = obs["is_new"].astype(bool)
    obs = obs.assign(
        fraction=(obs["ms_played"] / (obs["d"] * 60_000)).clip(upper=1),
        endplay=obs["reason_end"].eq("endplay"),
        y_after_no_skip=obs["y"].where(obs["prev_y"] == 0),
        y_known=obs["y"].where(~is_new),
        y_new=obs["y"].where(is_new),
    )
    return obs.groupby("arm").agg(
        plays=("y", "size"), labeled=("y", "count"), skip_rate=("y", "mean"),
        fraction_listened=("fraction", "mean"), endplay_share=("endplay", "mean"),
        skip_after_no_skip=("y_after_no_skip", "mean"), skip_known=("y_known", "mean"),
        skip_new=("y_new", "mean"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analysis of the experiment.")
    parser.add_argument("export_dir", help="folder of the new export of the streaming history")
    parser.add_argument(
        "--playlists", type=Path, default=EXPERIMENT_DIR, help="folder of the weekly playlists"
    )
    parser.add_argument(
        "--count-only", action="store_true", help="only count the plays, no outcomes by arm"
    )
    args = parser.parse_args()

    # 1. New history with the same cleaning, label and features as the model
    df = build_features(clean(load_raw(Path(args.export_dir))))
    playlists = load_playlists(args.playlists)
    df = add_arm(df, playlists)
    df["experimental"] = experiment_plays(df, playlists)
    plays = df[df["experimental"] & ~df["resume"]]
    labeled = plays[plays["labeled"]]

    # 2. Counts per weekly playlist, without outcomes: the only output allowed before the stop
    counts = plays.groupby("uploaded").agg(
        plays=("y", "size"), labeled=("labeled", "sum"), sessions=("session_id", "nunique")
    )
    counts.index = counts.index.tz_convert("Europe/Berlin").strftime("%Y-%m-%d")
    print(counts.to_string())
    print(f"Labeled experimental plays: {len(labeled):,} (stopping rule: {TARGET_PLAYS:,})")
    if args.count_only:
        sys.exit()

    # 3. Primary metric: skip rate per arm, difference and odds ratio, 95% bootstrap intervals
    y = labeled["y"].to_numpy()
    milp = labeled["arm"].eq("milp").to_numpy()
    session = labeled["session_id"].to_numpy()
    estimate = effect(y[~milp].mean(), y[milp].mean())
    table = pd.DataFrame({"estimate": estimate}).join(bootstrap_ci(y, milp, session, rng=SEED))
    print("\nPrimary metric (labeled experimental plays):")
    print(table.round(4).to_string())

    # 4. One-sided permutation test, labels permuted within each session
    observed, p_value = permutation_test(y, milp, session, rng=SEED)
    print(f"Permutation test: difference {observed:.4f}, one-sided p-value {p_value:.4f}")

    # 5. Secondary results, descriptive
    print("\nSecondary results:")
    print(secondary(plays).round(3).to_string())
