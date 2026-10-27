"""Load Spotify's extended streaming history, clean it and save it as a single file."""

from pathlib import Path

import pandas as pd

# Paths start from this file, so the script works from any folder
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "Spotify Extended Streaming History"
OUT_PATH = ROOT / "data" / "streams_clean.parquet"

FIRST_DAY = "2020-01-01"  # earlier years have too few plays
# Columns to keep, with shorter names
COLUMNS = {
    "ts": "ts",
    "spotify_track_uri": "track_uri",
    "master_metadata_track_name": "track",
    "master_metadata_album_artist_name": "artist",
    "master_metadata_album_album_name": "album",
    "ms_played": "ms_played",
    "reason_start": "reason_start",
    "reason_end": "reason_end",
    "shuffle": "shuffle",
    "offline": "offline",
    "incognito_mode": "incognito",
    "platform": "platform",
    "conn_country": "country",
}


def load_raw(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Read all audio files of the export into one table."""
    files = sorted(raw_dir.glob("Streaming_History_Audio_*.json"))
    print(f"Files found: {len(files)}")
    df = pd.concat([pd.read_json(f) for f in files], ignore_index=True)
    print(f"Rows loaded: {len(df)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Music plays since FIRST_DAY, without duplicates, with short column names, sorted by time."""
    # 1. Timestamp from text to date and time (UTC)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)

    # 2. Music only (drops podcasts, audiobooks and empty rows)
    n = len(df)
    df = df[df["spotify_track_uri"].notna()]
    print(f"Removed (not music): {n - len(df)}")

    # 3. Duplicates: same moment, same track, same time played
    n = len(df)
    df = df.drop_duplicates(subset=["ts", "spotify_track_uri", "ms_played"])
    print(f"Removed (duplicates): {n - len(df)}")

    # 4. Plays since FIRST_DAY
    n = len(df)
    df = df[df["ts"] >= pd.Timestamp(FIRST_DAY, tz="UTC")]
    print(f"Removed (before {FIRST_DAY[:4]}): {n - len(df)}")

    # 5. Useful columns only, renamed, in time order
    df = df[list(COLUMNS)].rename(columns=COLUMNS)
    return df.sort_values("ts").reset_index(drop=True)


if __name__ == "__main__":
    df = clean(load_raw())
    print(f"Final rows: {len(df)}")
    df.info()

    # Saved inside data/, which git ignores
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {OUT_PATH.relative_to(ROOT)}")
