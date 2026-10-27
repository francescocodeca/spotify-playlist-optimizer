"""Simulated export of the streaming history, to prepare the analysis before the real data.

The real history plus invented plays during the experiment: every day one session on the
active playlist (chain of skips, MILP tracks with a chosen odds ratio) and one session of free
listening. Written in the format of the Spotify export, so that the analysis reads it exactly as
it will read the real one. Never used for results.
From the terminal: python src/simulate_history.py [--odds-ratio 0.5] [--weeks 13]
"""

import argparse
import json
import shutil

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from experiment import plan_weeks
from features import ROOT
from load_data import RAW_DIR
from train import MODEL_PATH

# Simulated export (JSON files) and weekly playlists, inside data/ (ignored by git)
SIM_DIR = ROOT / "data" / "simulated"
FIRST_MONDAY = "2026-10-26"  # first upload of the experiment
UPLOAD_HOUR = 6  # upload time on Monday (German time), as in launchd
# Chain of skips as in the simulator of notebook 05: skip probability of the control arm after
# a non-skipped track, and shift of the logit after a skip
P_CONTROL = 0.035
SHIFT = 2.5
P_ENDPLAY = 0.04  # share of plays ended by choosing another track (censored)
P_LOGOUT = 0.02  # share of plays interrupted (censored)
PLAYLIST_TRACKS = (15, 25)  # tracks per daily session on the playlist (uniform)
FREE_TRACKS = 25  # tracks per daily session of free listening
SEED = 2026


def simulate_session(
    tracks: pd.DataFrame, start: pd.Timestamp, odds_ratio: float, rng: np.random.Generator,
) -> list[dict]:
    """Plays of one session, in the format of the export, one track after the other.

    tracks: columns track_uri, track, artist, duration_s and milp (True for the MILP arm, False for
    the control arm and for free listening).
    """
    rows = []
    t = start
    reason_start = "clickrow"
    skipped = False
    for track in tracks.itertuples():
        p = expit(logit(P_CONTROL) + np.log(odds_ratio) * track.milp + SHIFT * skipped)
        u = rng.random()
        if u < P_ENDPLAY:
            reason_end = "endplay"
        elif u < P_ENDPLAY + P_LOGOUT:
            reason_end = "logout"
        else:
            reason_end = "fwdbtn" if rng.random() < p else "trackdone"
        # Played to the end, or a random part of the track (most skips come early)
        full_ms = int(track.duration_s * 1000)
        ms = full_ms if reason_end == "trackdone" else int(full_ms * rng.beta(0.5, 2))
        t += pd.Timedelta(milliseconds=ms)
        rows.append({
            "ts": f"{t:%Y-%m-%dT%H:%M:%SZ}", "platform": "ios", "ms_played": ms,
            "conn_country": "DE", "master_metadata_track_name": track.track,
            "master_metadata_album_artist_name": track.artist,
            "master_metadata_album_album_name": None, "spotify_track_uri": track.track_uri,
            "reason_start": reason_start, "reason_end": reason_end, "shuffle": True,
            "skipped": reason_end == "fwdbtn", "offline": False, "incognito_mode": False,
        })
        skipped = reason_end == "fwdbtn"
        reason_start = reason_end if reason_end in ["trackdone", "fwdbtn"] else "playbtn"
        t += pd.Timedelta(seconds=1)
    return rows


def simulate_plays(
    playlists: pd.DataFrame, pool: pd.DataFrame, odds_ratio: float, rng: np.random.Generator,
) -> list[dict]:
    """Invented plays from the first upload to one week after the last one.

    Every day a session at 18:00 on the active playlist (a random subset of its tracks) and a
    session at 21:00 of free listening (random candidates). playlists: columns uploaded,
    track_uri and arm; pool: columns track_uri, track, artist and duration_s.
    """
    info = pool.set_index("track_uri")[["track", "artist", "duration_s"]]
    tracks = playlists.join(info, on="track_uri").assign(milp=lambda x: x["arm"] == "milp")
    uploads = tracks["uploaded"].drop_duplicates().sort_values()
    days = pd.date_range(uploads.iloc[0].normalize(), periods=7 * len(uploads), freq="D")

    rows = []
    for day in days:
        evening = day + pd.Timedelta(hours=18)
        active = tracks[tracks["uploaded"] == uploads[uploads <= evening].iloc[-1]]
        n = rng.integers(PLAYLIST_TRACKS[0], PLAYLIST_TRACKS[1] + 1)
        session = active.sample(n, random_state=rng)
        rows += simulate_session(session, evening, odds_ratio, rng)
        free = info.sample(FREE_TRACKS, random_state=rng).reset_index().assign(milp=False)
        rows += simulate_session(free, day + pd.Timedelta(hours=21), odds_ratio, rng)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated export of the streaming history.")
    parser.add_argument(
        "--odds-ratio", type=float, default=0.5, help="true effect of the MILP arm (1 = none)"
    )
    parser.add_argument("--weeks", type=int, default=13, help="weeks of the experiment")
    args = parser.parse_args()

    # 1. Weekly playlists, built as in the experiment, with an upload time
    streams = pd.read_parquet(ROOT / "data" / "streams_clean.parquet")
    pool = pd.read_parquet(ROOT / "data" / "pool.parquet")
    model = joblib.load(MODEL_PATH)
    mondays = pd.date_range(FIRST_MONDAY, periods=args.weeks, freq="7D")
    playlists = plan_weeks(streams, pool, model, mondays)
    upload = playlists["monday"] + pd.Timedelta(hours=UPLOAD_HOUR)
    playlists["uploaded"] = upload.dt.tz_localize("Europe/Berlin").dt.tz_convert("UTC")

    # 2. Invented plays
    rows = simulate_plays(playlists, pool, args.odds_ratio, np.random.default_rng(SEED))

    # 3. Export: the real JSON files plus one file of invented plays; one CSV per week
    export_dir = SIM_DIR / "export"
    playlist_dir = SIM_DIR / "experiment"
    for folder in [export_dir, playlist_dir]:
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    for f in RAW_DIR.glob("Streaming_History_Audio_*.json"):
        shutil.copy(f, export_dir)
    with open(export_dir / "Streaming_History_Audio_simulated.json", "w") as f:
        json.dump(rows, f)
    for monday, week in playlists.groupby("monday"):
        week.drop(columns="monday").to_csv(playlist_dir / f"{monday:%Y-%m-%d}.csv", index=False)
    print(f"{len(rows):,} invented plays, {args.weeks} weeks, odds ratio {args.odds_ratio}")
    print(f"Saved in {SIM_DIR.relative_to(ROOT)}")
