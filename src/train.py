"""Train the skip-prediction model (logistic regression) and predict the skip probability of
candidate tracks at a future moment.

Key idea for prediction: for each candidate track a "virtual row" is appended to the history
and the features are recomputed with the training code (features.build_features). The
prediction features are therefore computed exactly like the ones the model saw in training.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import ROOT, add_context, build_features

FEATURES = [
    "track_rate", "artist_rate", "global_rate_30d", "track_n_prev", "artist_n_prev",
    "log_days_since_last", "first_play", "hour", "weekend", "shuffle", "offline",
    "abroad", "device", "prev_y",
]
BOOL_COLS = ["first_play", "weekend", "shuffle", "offline", "abroad"]
# Fixed list and order: the model columns must not depend on which devices appear in the data
DEVICES = ["computer", "unknown", "speaker", "phone", "tv_console"]
MODEL_PATH = ROOT / "data" / "model.joblib"  # in data/, which git ignores


def model_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Rows to train and evaluate on: labeled, not resumes, started as in a playlist
    (after a finished track or after a skip).
    """
    mask = df["labeled"] & ~df["resume"] & df["reason_start"].isin(["trackdone", "fwdbtn"])
    return df[mask]


def to_Z(d: pd.DataFrame) -> pd.DataFrame:
    """Numeric matrix for the logistic regression."""
    Z = d[FEATURES].copy()
    for c in BOOL_COLS:
        Z[c] = Z[c].astype(int)
    Z["device"] = pd.Categorical(Z["device"], categories=DEVICES)
    Z = pd.get_dummies(Z, columns=["device"], dtype=float)  # one 0/1 column per device
    Z["prev_unknown"] = Z["prev_y"].isna().astype(float)  # previous track unknown
    Z["prev_y"] = Z["prev_y"].fillna(0)
    # First play: no previous play (already flagged by first_play)
    Z["log_days_since_last"] = Z["log_days_since_last"].fillna(0)
    # Hour on a circle: 23 and 0 are close
    Z["hour_sin"] = np.sin(2 * np.pi * Z["hour"] / 24)
    Z["hour_cos"] = np.cos(2 * np.pi * Z["hour"] / 24)
    return Z.drop(columns="hour")


def train(rows: pd.DataFrame):
    """Fit the standardized logistic regression on the given rows."""
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(to_Z(rows), rows["y"])
    return model


def candidate_features(
    streams: pd.DataFrame, track_uris, when, platform: str = "ios", country: str = "DE",
    shuffle: bool = True, offline: bool = False, asof=None,
) -> pd.DataFrame:
    """Features of candidate tracks for a playlist played at `when` (German time if naive).

    One virtual row per track is appended after the history and all features are recomputed.
    Virtual rows have no label, so they change neither the rates nor the counts of other tracks.

    History features (rates, counts, days since the last play, recent level) are computed at
    `asof` (tz-aware); by default just after the last play in `streams`. After that moment the
    history is unknown (for example during the experiment, because the Spotify export is not
    updated): computing them later would only empty the 30-day window and decay the rates
    artificially. Context features (hour, weekend, abroad) come from `when`.
    """
    when = pd.Timestamp(when)
    if when.tz is None:
        when = when.tz_localize("Europe/Berlin")
    when = when.tz_convert("UTC")
    last = streams["ts"].max()
    assert when > last, "the playlist must come after the last play in the history"
    asof = last + pd.Timedelta(seconds=1) if asof is None else pd.Timestamp(asof)
    track_uris = list(dict.fromkeys(track_uris))  # no duplicates, order kept

    info = (
        streams.drop_duplicates("track_uri", keep="last")
        .set_index("track_uri")
        .loc[track_uris, ["track", "artist", "album"]]
    )
    virtual = info.reset_index().assign(
        ts=asof,
        ms_played=0,
        reason_start="trackdone",  # as in a playlist; also prevents it from looking like a resume
        reason_end=None,  # unknown outcome -> no label
        shuffle=shuffle,
        offline=offline,
        incognito=False,
        platform=platform,
        country=country,
    )
    full = pd.concat([streams, virtual[streams.columns]], ignore_index=True)
    full["ts"] = full["ts"].astype(streams["ts"].dtype)
    feats = build_features(full).iloc[len(streams):].reset_index(drop=True)

    # Context of the playlist moment, with the same code as in training
    context = add_context(pd.DataFrame({"ts": [when], "country": [country]}))
    for c in ["hour", "weekend", "abroad"]:
        feats[c] = context[c].iloc[0]
    return feats


def predict_candidates(model, cand: pd.DataFrame) -> pd.DataFrame:
    """Two skip probabilities per candidate: p0 if the previous track was not skipped,
    p1 if it was (the two conditional probabilities of the chain).
    """
    out = cand[["track_uri", "track", "artist"]].copy()
    for v in [0, 1]:
        out[f"p{v}"] = model.predict_proba(to_Z(cand.assign(prev_y=float(v))))[:, 1]
    return out


if __name__ == "__main__":
    df = pd.read_parquet(ROOT / "data" / "features.parquet")
    rows = model_rows(df)

    # 1. Consistency with notebook 02: same split, same AUC (0.836)
    cut = pd.Timestamp("2025-07-01", tz="UTC")
    tr, te = rows[rows["ts"] < cut], rows[rows["ts"] >= cut]
    auc = roc_auc_score(te["y"], train(tr).predict_proba(to_Z(te))[:, 1])
    print(f"Test AUC (split 2025-07-01): {auc:.3f}")
    assert abs(auc - 0.836) < 0.002, "result differs from notebook 02"

    # 2. Final model: all available data
    model = train(rows)
    joblib.dump(model, MODEL_PATH)
    print(f"Model trained on {len(rows)} rows, saved in {MODEL_PATH.relative_to(ROOT)}")

    # 3. Virtual-row check: rebuild a real row with the virtual method (history features at the
    #    moment of the row) and check that the features equal those in features.parquet
    streams = pd.read_parquet(ROOT / "data" / "streams_clean.parquet")
    r = rows.iloc[-1]
    check = candidate_features(
        streams[streams["ts"] < r["ts"]], [r["track_uri"]], r["ts"], platform=r["platform"],
        country=r["country"], shuffle=r["shuffle"], offline=r["offline"], asof=r["ts"],
    ).iloc[0]
    for c in FEATURES:
        if c == "prev_y":
            continue
        a, b = check[c], r[c]
        if c == "device":
            same = a == b
        else:
            same = (pd.isna(a) and pd.isna(b)) or np.isclose(float(a), float(b))
        assert same, f"feature {c} differs: {a} vs {b}"
    print("Virtual row: features identical to the training ones")

    # 4. Example: the 30 most played tracks of the last 90 days, tomorrow at 8:00, phone, shuffle
    recent = streams[streams["ts"] >= streams["ts"].max() - pd.Timedelta(days=90)]
    top = recent["track_uri"].value_counts().index[:30]
    tomorrow = (pd.Timestamp.now(tz="Europe/Berlin") + pd.Timedelta(days=1)).normalize()
    when = tomorrow + pd.Timedelta(hours=8)
    pred = predict_candidates(model, candidate_features(streams, top, when))
    print(f"\nPredictions for {when:%Y-%m-%d %H:%M} (first 10 by increasing p0):")
    print(pred.sort_values("p0").head(10).round(3).to_string(index=False))
    print(f"\nmean p0 {pred['p0'].mean():.3f}, mean p1 {pred['p1'].mean():.3f}")
