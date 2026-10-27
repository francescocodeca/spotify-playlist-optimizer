"""Candidate tracks for the playlists, with the estimated duration of each track.

Duration: Spotify does not provide it. When a track ends by itself (reason_end = trackdone),
ms_played is the whole duration, so the median of ms_played over the trackdone plays is used.
"""

import pandas as pd

from features import ROOT

MIN_DURATION_S = 30  # shorter estimates are dirty data (medians at 0 or almost)
MIN_PLAYS = 3  # "my" track: played at least 3 times (it also has a useful skip history)
ACTIVE_DAYS = 365  # ...and at least once in the last year (excludes abandoned tracks)
NEW_DAYS = 60  # "new" track: first play in the last 60 days (enters with fewer plays too)


def track_durations(df: pd.DataFrame) -> pd.Series:
    """Estimated duration in seconds per track: median of ms_played over the trackdone plays.

    Tracks never finished by themselves have no estimate and stay out.
    """
    done = df[df["reason_end"] == "trackdone"]
    dur = done.groupby("track_uri")["ms_played"].median() / 1000
    return dur[dur >= MIN_DURATION_S].rename("duration_s")


def candidate_pool(df: pd.DataFrame) -> pd.DataFrame:
    """Candidate tracks: known duration and (mine and active, or new).

    `df` = features.parquet (the resume column is needed: resumes are not new plays).
    """
    now = df["ts"].max()
    plays = df[~df["resume"]]
    g = plays.groupby("track_uri").agg(
        track=("track", "last"), artist=("artist", "last"), plays=("ts", "size"),
        first_ts=("ts", "min"), last_ts=("ts", "max"),
    )
    g = g.join(track_durations(df), how="inner")  # inner: only tracks with an estimated duration
    g["is_new"] = g["first_ts"] >= now - pd.Timedelta(days=NEW_DAYS)
    active = g["last_ts"] >= now - pd.Timedelta(days=ACTIVE_DAYS)
    pool = g[(active & (g["plays"] >= MIN_PLAYS)) | g["is_new"]]
    return pool.reset_index()


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data" / "features.parquet")

    dur = track_durations(df)
    print(f"Tracks with an estimated duration: {len(dur)}")
    print((dur / 60).rename("duration_min").describe().round(2), "\n")

    # Check: over the trackdone plays the duration of a track is almost always the same
    done = df[df["reason_end"] == "trackdone"]
    spread = done.groupby("track_uri")["ms_played"].agg(lambda x: x.max() - x.min()) / 1000
    print(f"Tracks whose trackdone plays are all within 2 s: {(spread <= 2).mean():.1%}\n")

    pool = candidate_pool(df)
    print(
        f"Candidates: {len(pool)} tracks, {pool['artist'].nunique()} artists, "
        f"new: {pool['is_new'].sum()}"
    )
    print(f"Total duration: {pool['duration_s'].sum() / 3600:.0f} hours")
    print("Tracks per artist (candidates):")
    print(pool["artist"].value_counts().describe().round(1))

    assert pool["track_uri"].is_unique
    assert (pool["duration_s"] >= MIN_DURATION_S).all()
    pool.to_parquet(ROOT / "data" / "pool.parquet", index=False)
    print("\nSaved data/pool.parquet")
