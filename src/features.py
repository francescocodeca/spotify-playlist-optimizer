"""Build the features of the skip-prediction model.

Golden rule: every feature of a row uses only EARLIER rows (no information from the future).
The DataFrame must be sorted by `ts`.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Project folder (src/ is a subfolder): paths work from any working directory
ROOT = Path(__file__).resolve().parent.parent

# End reasons with an observed outcome; all the others are interruptions (censored outcome)
OBSERVED = ["trackdone", "fwdbtn", "endplay", "backbtn"]
# End reasons that give the label: played to the end or skipped forward
LABEL_ENDS = ["trackdone", "fwdbtn"]
# "Technical" start reasons: the track restarts without a choice (app reopened, device change, ...)
TECHNICAL_START = ["appload", "remote", "playbtn", "trackerror", "unknown"]

ALPHA = 5  # shrinkage strength: "pseudo-plays" at the prior rate (notebook 02)
H_ARTIST = 90  # half-life (days) of the weights in the artist rate (notebook 02)
H_TRACK = 365  # half-life (days) of the weights in the track rate (notebook 02)
SESSION_GAP_MIN = 30  # pause (minutes) that starts a new session (notebook 01)
HOME_COUNTRIES = ["IT", "DE"]


def add_label(df: pd.DataFrame) -> pd.DataFrame:
    """Label `y`: 1 = skipped with the forward button (fwdbtn), 0 = played to the end (trackdone).

    NaN for everything else: interruptions, and endplay/backbtn (I chose another track before it
    was clear whether I would have skipped this one). For this question they are censored.
    """
    df["labeled"] = df["reason_end"].isin(LABEL_ENDS)
    df["y"] = (df["reason_end"] == "fwdbtn").astype(float).where(df["labeled"])
    return df


def add_resume_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Resume = same track as the previous row, technical start, previous row interrupted.

    It is the same play split into two rows.
    """
    same_track = df["track_uri"].eq(df["track_uri"].shift())
    technical = df["reason_start"].isin(TECHNICAL_START)
    prev_interrupted = ~df["reason_end"].shift().isin(OBSERVED)
    df["resume"] = same_track & technical & prev_interrupted
    return df


def platform_group(p: str) -> str:
    """Reduce Spotify's raw platform string to a device category."""
    p = p.lower()
    if p == "not_applicable":
        return "unknown"
    if "echo" in p or "amazon" in p:
        return "speaker"
    if any(k in p for k in ["ps4", "playstation", "android_tv", "tv"]):
        return "tv_console"
    if p.startswith(("ios", "android")):
        return "phone"
    if p.startswith(("osx", "os x", "windows", "web")):
        return "computer"
    return "other"


def add_track_history(df: pd.DataFrame) -> pd.DataFrame:
    """History of the track before the current row: previous plays, days since the last play,
    first play. Resumes do not count as new plays.
    """
    df["new_play"] = (~df["resume"]).astype(int)

    # Previous plays: cumulative sum of the track's new plays, minus the current row
    df["track_plays_prev"] = df.groupby("track_uri")["new_play"].cumsum() - df["new_play"]

    # Time of the last new play BEFORE this row:
    # 1) keep ts only on rows that are new plays (NaT on resumes)
    # 2) shift by one row within each track (so the current row does not see itself)
    # 3) forward-fill the gaps left by resumes
    last_new = df["ts"].where(df["new_play"] == 1)
    prev_new = last_new.groupby(df["track_uri"]).shift()
    prev_new = prev_new.groupby(df["track_uri"]).ffill()
    df["days_since_last"] = (df["ts"] - prev_new).dt.total_seconds() / 86400
    df["log_days_since_last"] = np.log1p(df["days_since_last"])  # NaN on the first play

    df["first_play"] = df["track_plays_prev"] == 0
    return df


def past_rate(
    df: pd.DataFrame, key: str, prior: pd.Series, alpha: float = ALPHA, h: float = np.inf
) -> pd.DataFrame:
    """Past skip rate per group (`key` = track or artist), from earlier rows only.

    Shrinkage towards `prior`: (weighted past skips + alpha * prior) / (weighted past outcomes
    + alpha). Each past outcome weighs 2^(-Δt/h), Δt in days: after h days it counts half;
    h = inf gives all weights 1.
    Returns two columns: number of past outcomes (not weighted) and shrunk rate.
    """
    y0 = df["y"].fillna(0)  # unlabeled rows add no skips...
    lab = df["labeled"].astype(float)  # ...and do not count as outcomes
    # Plain count, used as a feature
    n_prev = (lab.groupby(df[key]).cumsum() - lab).astype(int)

    # Weights: 2^(-(t - t_i)/h) = 2^(t_i/h) / 2^(t/h). Each outcome is multiplied by
    # g = 2^(days/h), summed cumulatively per group (minus the current row) and divided by g of
    # the current row: one pass instead of a new sum for every row. With h >= 20 days g < 1e40.
    if np.isfinite(h):
        days = (df["ts"] - df["ts"].iloc[0]).dt.total_seconds() / 86400
        g = 2.0 ** (days / h)
    else:
        g = pd.Series(1.0, index=df.index)
    n_w = ((lab * g).groupby(df[key]).cumsum() - lab * g) / g
    s_w = ((y0 * g).groupby(df[key]).cumsum() - y0 * g) / g
    rate = (s_w + alpha * prior) / (n_w + alpha)
    return pd.DataFrame({"n": n_prev, "rate": rate})


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Past skip rates: global, artist, track.

    Hierarchy: the track rate is shrunk towards its artist, the artist rate towards the recent
    global level.
    """
    lab = df["labeled"].astype(int)
    y0 = df["y"].fillna(0)

    # Global, whole past history. On the first row there is no past: 0.5 (neutral prior).
    df["global_rate"] = ((y0.cumsum() - y0) / (lab.cumsum() - lab)).fillna(0.5)

    # Recent global level (last 30 days, current row excluded): follows the drift.
    # mean() over the window ignores NaN, so it uses labeled rows only.
    # In the very first days the window is empty: global rate instead.
    df["global_rate_30d"] = (
        df.rolling("30D", on="ts", closed="left")["y"].mean().fillna(df["global_rate"])
    )

    # The artist prior is the recent level, not the whole history: with the drift it works
    # better (the whole-history rate is pulled up by 2020-2021).
    # Decaying weights: the relationship with an artist goes in phases (short half-life),
    # the taste for a single track is stable (long half-life). Both chosen with the backtest.
    artist = past_rate(df, "artist", prior=df["global_rate_30d"], h=H_ARTIST)
    df["artist_n_prev"], df["artist_rate"] = artist["n"], artist["rate"]

    track = past_rate(df, "track_uri", prior=df["artist_rate"], h=H_TRACK)
    df["track_n_prev"], df["track_rate"] = track["n"], track["rate"]
    return df


def add_context(df: pd.DataFrame) -> pd.DataFrame:
    """Listening context: local hour, weekend, abroad."""
    local = df["ts"].dt.tz_convert("Europe/Rome")  # same time zone as Berlin
    df["hour"] = local.dt.hour
    df["weekend"] = local.dt.day_of_week >= 5
    df["abroad"] = ~df["country"].isin(HOME_COUNTRIES)
    return df


def add_session(df: pd.DataFrame) -> pd.DataFrame:
    """Sessions (pause > SESSION_GAP_MIN minutes) and skip of the previous track in the session."""
    start = df["ts"] - pd.to_timedelta(df["ms_played"], unit="ms")
    gap_min = (start - df["ts"].shift()).dt.total_seconds() / 60
    df["session_id"] = (gap_min > SESSION_GAP_MIN).cumsum()
    # NaN for the first track of a session or after an unlabeled track
    df["prev_y"] = df.groupby("session_id")["y"].shift()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all the steps in order. Each step adds columns to df."""
    assert df["ts"].is_monotonic_increasing, "df must be sorted by time"
    df = add_label(df)
    df = add_resume_flag(df)
    df["device"] = df["platform"].map(platform_group)
    df = add_track_history(df)
    df = add_rates(df)
    df = add_context(df)
    df = add_session(df)
    return df


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    df = pd.read_parquet(ROOT / "data" / "streams_clean.parquet")
    df = build_features(df)

    # 1. Quick checks
    print(
        f"Rows: {len(df)}, labeled: {df['labeled'].mean():.1%}, "
        f"skip rate (y): {df['y'].mean():.1%}"
    )
    print(f"Resumes: {df['resume'].sum()} ({df['resume'].mean():.1%})")
    print(df["device"].value_counts(), "\n")
    print(df[["track_plays_prev", "days_since_last"]].describe().round(2))

    # 2. Consistency: a first play has no previous play, the others do
    assert df.loc[df["first_play"], "days_since_last"].isna().all()
    assert df.loc[~df["first_play"], "days_since_last"].notna().all()
    assert (df["days_since_last"].dropna() >= 0).all()

    # 3. Rates: all between 0 and 1 and defined on every row
    rates = ["global_rate", "global_rate_30d", "artist_rate", "track_rate"]
    for c in rates:
        assert df[c].between(0, 1).all(), c
    print(df[rates].describe().round(3), "\n")

    # 4. Quality: how well past rates separate skips from non-skips (labeled rows from 07/2021)
    ev = df[df["labeled"] & (df["ts"] >= pd.Timestamp("2021-07-01", tz="UTC"))]
    for c in ["global_rate_30d", "artist_rate", "track_rate"]:
        print(f"AUC {c}: {roc_auc_score(ev['y'], ev[c]):.3f}")
    print(ev.groupby("prev_y", dropna=False)["y"].agg(rate="mean", n="size").round(3), "\n")

    df.to_parquet(ROOT / "data" / "features.parquet", index=False)
    print("Saved data/features.parquet")
