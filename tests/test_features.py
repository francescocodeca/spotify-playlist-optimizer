"""Tests of the features on small hand-made histories."""

import numpy as np
import pandas as pd
import pytest

from features import (
    add_label, add_resume_flag, add_session, add_track_history, build_features, past_rate,
    platform_group,
)

START = pd.Timestamp("2026-01-05 08:00", tz="UTC")


def history(rows: list[tuple]) -> pd.DataFrame:
    """History in the format of streams_clean, one row per tuple.

    Tuple: (minutes from START at the end of the play, track, reason_start, reason_end).
    Every play lasts one minute; the artist is the first letter of the track.
    """
    df = pd.DataFrame(rows, columns=["minute", "track_uri", "reason_start", "reason_end"])
    df["ts"] = START + pd.to_timedelta(df.pop("minute"), unit="min")
    df["artist"] = df["track_uri"].str[0]
    df["ms_played"] = 60_000
    df["shuffle"] = True
    df["offline"] = False
    df["platform"] = "iOS 18.0 (iPhone16,2)"
    df["country"] = "DE"
    return df


def test_label():
    """fwdbtn = 1, trackdone = 0, everything else unlabeled."""
    df = history([
        (1, "a1", "trackdone", "trackdone"), (2, "a2", "trackdone", "fwdbtn"),
        (3, "a3", "fwdbtn", "endplay"), (4, "a4", "clickrow", "logout"),
    ])
    df = add_label(df)
    assert df["labeled"].tolist() == [True, True, False, False]
    assert df["y"].tolist()[:2] == [0.0, 1.0]
    assert df["y"].iloc[2:].isna().all()


def test_resume_flag():
    """Only an interrupted track that restarts by itself is a resume."""
    df = history([
        # Interrupted, then the same track restarts by itself: a resume
        (1, "a1", "trackdone", "logout"), (2, "a1", "appload", "trackdone"),
        # Played to the end, then started again: a new play
        (3, "a1", "playbtn", "trackdone"),
        # Interrupted, then another track: a new play
        (4, "b1", "trackdone", "logout"), (5, "c1", "appload", "trackdone"),
    ])
    df = add_resume_flag(df)
    assert df["resume"].tolist() == [False, True, False, False, False]


@pytest.mark.parametrize(
    "platform, device",
    [
        ("iOS 18.0 (iPhone16,2)", "phone"),
        ("not_applicable", "unknown"),
        # Echo runs Android: the speaker rule must come before the phone rule
        ("Android OS 9 API 28 (Amazon, Echo)", "speaker"),
        ("Android_TV (Sony, BRAVIA)", "tv_console"),
        ("OS X 14.1.0 [arm 2]", "computer"),
        ("web_player osx 14.1.0", "computer"),
        ("something new", "other"),
    ],
)
def test_platform_group(platform, device):
    """Raw platform strings mapped to device categories."""
    assert platform_group(platform) == device


def test_track_history_ignores_resumes():
    """A resume is not a new play of the track."""
    df = history([
        (1, "a1", "trackdone", "logout"), (2, "a1", "appload", "trackdone"),
        (3 * 1440, "a1", "trackdone", "trackdone"),
    ])
    df = add_track_history(add_resume_flag(df))
    assert df["first_play"].tolist() == [True, False, False]
    # The resume is not a new play: the third row has one previous play...
    assert df["track_plays_prev"].iloc[2] == 1
    # ...and the days since the last play start from the first row, not from the resume
    assert df["days_since_last"].iloc[2] == pytest.approx((3 * 1440 - 1) / 1440)
    assert np.isnan(df["days_since_last"].iloc[0])


def test_past_rate_without_decay():
    """Shrunk rate from the earlier labeled rows only."""
    df = history([
        (1, "a1", "trackdone", "fwdbtn"), (2, "a1", "trackdone", "trackdone"),
        (3, "a1", "trackdone", "endplay"), (4, "a1", "trackdone", "trackdone"),
    ])
    df = add_label(df)
    out = past_rate(df, "track_uri", prior=pd.Series(0.5, index=df.index), alpha=1)
    # (past skips + alpha * prior) / (past outcomes + alpha); endplay is not an outcome
    assert out["n"].tolist() == [0, 1, 2, 2]
    assert out["rate"].tolist() == pytest.approx([0.5, 1.5 / 2, 1.5 / 3, 1.5 / 3])


def test_past_rate_with_decay():
    """After h days a past outcome weighs one half."""
    h = 30
    df = history([(1, "a1", "trackdone", "fwdbtn"), (1 + h * 1440, "a1", "trackdone", "trackdone")])
    df = add_label(df)
    out = past_rate(df, "track_uri", prior=pd.Series(0.0, index=df.index), alpha=1, h=h)
    # After h days the skip weighs one half: 0.5 / (0.5 + 1); the count is not weighted
    assert out["rate"].iloc[1] == pytest.approx(0.5 / 1.5)
    assert out["n"].iloc[1] == 1


def test_past_rate_groups_are_separate():
    """The rate of a track uses only the plays of that track."""
    df = history([
        (1, "a1", "trackdone", "fwdbtn"), (2, "b1", "trackdone", "trackdone"),
        (3, "a1", "trackdone", "trackdone"), (4, "b1", "trackdone", "trackdone"),
    ])
    df = add_label(df)
    out = past_rate(df, "track_uri", prior=pd.Series(0.5, index=df.index), alpha=1)
    assert out["rate"].tolist() == pytest.approx([0.5, 0.5, 1.5 / 2, 0.5 / 2])


def test_sessions_and_previous_skip():
    """A pause of more than 30 minutes starts a new session."""
    df = history([
        (1, "a1", "trackdone", "fwdbtn"), (2, "a2", "fwdbtn", "trackdone"),
        # 40 minutes of pause: new session
        (43, "a3", "trackdone", "trackdone"), (44, "a4", "trackdone", "fwdbtn"),
    ])
    df = add_session(add_label(df))
    assert df["session_id"].tolist() == [0, 0, 1, 1]
    # Skip of the previous track in the same session; NaN at the start of each session
    assert df["prev_y"].iloc[1] == 1.0
    assert df["prev_y"].iloc[3] == 0.0
    assert df["prev_y"].iloc[[0, 2]].isna().all()


def random_rows(n: int, seed: int = 0) -> list[tuple]:
    """n random plays of 15 tracks of 3 artists, with pauses of 1 to 120 minutes."""
    rng = np.random.default_rng(seed)
    return [
        (
            int(m), f"{rng.choice(list("abc"))}{rng.integers(5)}",
            rng.choice(["trackdone", "fwdbtn", "appload"]),
            rng.choice(["trackdone", "fwdbtn", "endplay", "logout"], p=[0.5, 0.3, 0.1, 0.1]),
        )
        for m in np.cumsum(rng.integers(1, 120, size=n))
    ]


def test_no_information_from_the_future():
    """Golden rule: adding later plays does not change the features of the earlier rows."""
    rows = random_rows(200)
    full = build_features(history(rows))
    past = build_features(history(rows[:150]))
    pd.testing.assert_frame_equal(full.iloc[:150], past)


def test_no_information_from_the_outcome():
    """A row does not see its own outcome: changing it changes only the label of that row."""
    rows = random_rows(150)
    last = rows[-1]
    flipped = "fwdbtn" if last[3] == "trackdone" else "trackdone"
    changed = rows[:-1] + [last[:3] + (flipped,)]
    a = build_features(history(rows))
    b = build_features(history(changed))
    label = ["reason_end", "labeled", "y"]
    pd.testing.assert_frame_equal(a.drop(columns=label), b.drop(columns=label))
