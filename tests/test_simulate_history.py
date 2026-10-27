"""Test of the simulated export: the analysis finds every simulated session on the playlist."""

import numpy as np
import pandas as pd

from analysis import experiment_plays
from features import build_features
from load_data import clean
from simulate_history import simulate_plays


def test_simulated_sessions_are_found():
    """Plays of the daily sessions on the playlist are experimental, free listening is not."""
    week_1 = pd.Timestamp("2026-11-02 05:00", tz="UTC")
    playlists = pd.DataFrame({
        "uploaded": [week_1] * 30 + [week_1 + pd.Timedelta(days=7)] * 30,
        "track_uri": [f"p{i}" for i in range(60)],
        "arm": ["milp", "control"] * 30,
    })
    # Many more candidates than playlist tracks: free listening hardly ever meets the playlist
    uris = [f"p{i}" for i in range(60)] + [f"f{i}" for i in range(2000)]
    pool = pd.DataFrame({"track_uri": uris, "track": "t", "artist": "a", "duration_s": 180})
    rows = simulate_plays(playlists, pool, odds_ratio=0.5, rng=np.random.default_rng(0))

    df = build_features(clean(pd.DataFrame(rows)))
    found = experiment_plays(df, playlists)
    # Sessions on the playlist start at 18:00 and end before 21:00, when free listening starts
    assert found.tolist() == (df["ts"].dt.hour < 21).tolist()
