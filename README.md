# Optimized playlists from my Spotify listening data

Work in progress. The project follows the cycle *predict, decide, verify* on more than six years of my own listening history (about 111,000 music plays since 2020, from Spotify's extended streaming history).

1. **Predict:** a logistic regression estimates the probability that I skip a track (forward button) in a given context, from the history of the track and of the artist, the recent skip level, the context and the previous skip. Validated over time only (backtest with 7 cuts).
2. **Decide:** a MILP (PuLP + HiGHS) chooses the tracks of a playlist of given length: maximum expected minutes listened, with constraints on duration, tracks per artist, share of new tracks and long tracks.
3. **Verify:** an experiment on myself, optimized tracks against tracks chosen without the predictions, mixed in the same playlist (interleaving). Running since 2026-10-26, see below.

## Results so far

- Model: AUC 0.836 on the test period (from 2025-07); 0.726 against 0.713 for the track's own skip rate after a non-skipped track, the case the playlist uses. The gain comes from the quality of the track history (exponential decay), not from model complexity.
- Playlist: predicted skip probability 0.041 against 0.081 for random playlists from the candidates. Many tracks are tied at the top, so the MILP separates good tracks from bad ones but chooses almost at random among the good ones.
- Extensions tested and dropped with data: optimizing the order (no pair effect in the data, notebook 04) and a joint MILP over a week (same value as one playlist per day with a rotation filter, much slower because of symmetry, notebook 03).
- Not verified yet: predicted values are not real outcomes; the A/B test will measure them.

## The experiment (running)

**Question.** Are the tracks chosen by the MILP skipped less often than tracks chosen without the predictions, under the same constraints? One-sided test, pre-registered in [`docs/experiment_protocol.md`](docs/experiment_protocol.md) before the first upload.

**Design: interleaving**, the method used to compare recommender systems.

- One private Spotify playlist, replaced every Monday: 300 minutes of MILP tracks and 300 minutes of control tracks, mixed in random order. The arm of each track is saved but not looked at before the analysis.
- Control arm: the same MILP with a random value per minute instead of the predictions, so the two arms have the same duration, artists per playlist, share of new tracks and long tracks. They differ only in the use of the predictions.
- Listening is free (any time, any amount). While the playlist plays: normal shuffle, no queue, no click on a specific track.
- Why not one playlist per day, alternating the arms: plays in the same session and day are correlated (intraclass correlation 0.19 within a session), so 40 plays of one day are worth only 7–13 independent plays (design effect 3.0–5.5). With interleaving both arms share every session, and the session effect cancels out.

**Analysis.**

- Experimental plays are found in the streaming history, which does not record the playlist: a play of a track of the active playlist, inside a run of at least 3 such plays in the same session. On ten months of 2026 listening this rule finds 0.14 plays per week by chance.
- Primary metric: skip rate per arm, with difference and odds ratio, and 95% confidence intervals from a bootstrap over sessions.
- Test: permutation test with the arm labels permuted within each session (type I error 0.038 at the nominal 0.05 in simulation).

**Power** (simulation calibrated on my recent listening: skip chains, day and session effects):

| Labeled experimental plays | 500 | 1,200 | 1,700 |
|---|---|---|---|
| odds ratio 0.5 (the effect predicted by the model) | 0.33 | 0.65 | 0.80 |
| odds ratio 0.7 | 0.11 | 0.31 | 0.42 |

Over the 23 planned weeks the predicted contrast stays stable: the mean predicted skip probability of the MILP arm against the control arm has an odds ratio of 0.51 on average (0.46–0.60), with a 4-week rotation of the tracks.

**Status.** Started on 2026-10-26. Stop at 1,700 labeled experimental plays or on 2027-03-31, whichever comes first; then a new export of the history and `python src/analysis.py`. With a smaller effect than predicted, the result will be an estimate with a confidence interval rather than a significant test.

## Repository

| Path | Content |
|---|---|
| `src/load_data.py` | Loads and cleans the raw export into `data/streams_clean.parquet` |
| `src/features.py` | Label (forward skip against completed play) and features, built only from earlier rows |
| `src/train.py` | Final model; predictions for candidate tracks at a future moment ("virtual rows") |
| `src/candidates.py` | Estimated track durations and candidate tracks |
| `src/playlist.py` | MILP version 1 and the daily playlist with rotation |
| `src/experiment.py` | Weekly playlist of the experiment (both arms) and upload to Spotify |
| `src/spotify_client.py` | Spotify Web API client (OAuth) and the private playlist "Experiment" |
| `src/analysis.py` | Analysis of the experiment: experimental plays, effect with confidence intervals, permutation test |
| `docs/experiment_protocol.md` | Pre-registered protocol of the experiment |
| `notebooks/01_exploration.ipynb` | Exploration: context, track history, sessions, drift, label definition |
| `notebooks/02_model.ipynb` | Baselines, models, calibration, cold start, backtest, decay of the rates |
| `notebooks/03_milp.ipynb` | MILP: solution, long tracks, greedy and random comparisons, LP relaxation, sensitivity, rotation |
| `notebooks/04_order.ipynb` | Does the order of the tracks matter? |
| `notebooks/05_experiment_design.ipynb` | Design of the experiment: correlation, simulator, power, identification of the plays |

## Reproduce

```
conda env create -f environment.yml
conda activate spotify
python src/load_data.py
python src/features.py
python src/train.py
python src/candidates.py
python src/playlist.py "2026-10-26 08:00" --minutes 45
python src/experiment.py 2026-11-02
python src/analysis.py "<folder of a newer export>" --count-only
```

The raw data are personal and are not included: the scripts expect Spotify's "Extended streaming history" in `data/Spotify Extended Streaming History/`.
