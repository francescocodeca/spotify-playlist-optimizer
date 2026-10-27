"""Connection to the Spotify Web API (spotipy) and the experiment playlist.

Keys in .env (ignored by git): SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI.
From the terminal (connection test): python src/spotify_client.py
"""

from dotenv import load_dotenv
import pandas as pd
import spotipy
from spotipy.cache_handler import CacheFileHandler
from spotipy.oauth2 import SpotifyOAuth

from features import ROOT

# Permission requested: create and edit my private playlists
SCOPE = "playlist-modify-private"
# Access token saved after the first login (in data/, ignored by git)
TOKEN_CACHE = ROOT / "data" / ".spotify_token"
# ID of the experiment playlist, written when the playlist is created
PLAYLIST_ID_PATH = ROOT / "data" / "experiment_playlist_id.txt"
PLAYLIST_NAME = "Experiment"


def get_client() -> spotipy.Spotify:
    """Spotify client authorized on my account (the first time it opens the browser)."""
    load_dotenv(ROOT / ".env")
    auth = SpotifyOAuth(scope=SCOPE, cache_handler=CacheFileHandler(cache_path=str(TOKEN_CACHE)))
    return spotipy.Spotify(auth_manager=auth)


def experiment_playlist(sp: spotipy.Spotify) -> str:
    """ID of the experiment playlist; creates it (private) the first time."""
    if PLAYLIST_ID_PATH.exists():
        return PLAYLIST_ID_PATH.read_text().strip()
    pl = sp.current_user_playlist_create(
        PLAYLIST_NAME, public=False, description="Personal experiment, updated by a script"
    )
    PLAYLIST_ID_PATH.write_text(pl["id"])
    return pl["id"]


if __name__ == "__main__":
    sp = get_client()

    # 1. Login check
    print(f"Connected as {sp.current_user()['display_name']}")

    # 2. Experiment playlist (created the first time)
    playlist_id = experiment_playlist(sp)

    # 3. Writing check: the playlist gets one candidate track
    pool = pd.read_parquet(ROOT / "data" / "pool.parquet")
    sp.playlist_replace_items(playlist_id, pool["track_uri"].head(1).tolist())
    print(f"Playlist '{PLAYLIST_NAME}': {sp.playlist_items(playlist_id)['total']} track(s)")
