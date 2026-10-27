# Experiment protocol: optimized vs. control playlist (interleaving)

Pre-registered: this file is committed before the first experimental playlist is uploaded.
Any later change is listed in "Deviations", with its date and reason.

## Question
Are the tracks chosen by the MILP (version 1, using the skip predictions) skipped less often than
tracks chosen without the predictions, under the same constraints?

- H0: the skip probability is the same for the two arms.
- H1: the skip probability is lower for the MILP arm (one-sided).

## Design
Interleaving within a single playlist, so that day, mood and session affect both arms equally.

- One private Spotify playlist, "Experiment", replaced every Monday by `src/experiment.py`.
- Each week: 300 minutes of MILP tracks and 300 minutes of control tracks (about 190 tracks).
- MILP arm: version 1 of the MILP (`src/playlist.py`, parameters in `PARAMS`), predictions for
  Monday 8:00, features at the end of the history (last play: 2026-10-20).
- Control arm: the same MILP on the remaining candidates, with a random value per minute: same
  duration, at most 2 tracks per artist, same share of new tracks, same limit on long tracks.
- A track enters the playlist at most once every 4 weeks (both arms).
- Order: random. The random generator is seeded with the date of the Monday (reproducible).
- Blind: the arm of each track is saved in `data/experiment/` and not looked at before the
  analysis.

## Listening rules
- Listening is free: any time, any amount, other music outside the playlist.
- While the playlist plays: normal shuffle (not Smart Shuffle), skip as usual, no queue, no click
  on a specific track, no changes to the playlist. To choose a track, leave the playlist.

## Data
Spotify Extended Streaming History, requested after the end of the experiment.

- Experimental play: a row whose track is in the playlist active at its timestamp (between two
  uploads), inside a run of at least 3 consecutive such rows in the same session (gap <= 30 min).
  The history does not record the playlist; on 2026 data this rule finds about 0.05 runs per week
  by chance in free listening.
- Label as in `src/features.py`: skip = `fwdbtn` (1), `trackdone` (0), everything else censored.
  Resumed plays (`add_resume_flag`) are excluded.

## Analysis
- Primary metric: skip rate per arm over the labeled experimental plays. Reported as rates,
  difference and odds ratio, with 95% confidence intervals (bootstrap over sessions).
- Test: one-sided permutation test, arm labels permuted within each session (10,000
  permutations), alpha = 0.05.
- Secondary (descriptive, no test): fraction of the track listened per arm; `endplay` rate per
  arm; skip rate after a non-skipped track; results for known tracks and new tracks separately.

## Sample size and stopping rule
- Stop at 1,700 labeled experimental plays or on 2027-03-31, whichever comes first.
- No look at the outcomes by arm before the stop. An intermediate data request may be used only
  to count the experimental plays and check the identification rule.
- Power (simulation with skip chains and session effects, recent skip rate 7%): about 0.8 at
  1,700 plays if the effect is the one predicted by the model (odds ratio 0.5, skip rate 8% ->
  4%); about 0.4 with an odds ratio of 0.7. With a smaller effect the result is an estimate with
  a confidence interval rather than a significant test.

## Known limits
- Blinding is partial: favourite tracks can be recognized.
- Predictions use the history up to 2026-10-20: wear during the experiment is not seen by the
  model, limited by the 4-week rotation.
- The MILP maximizes expected minutes listened; the test measures the skip rate. Close, not
  identical.
- `endplay` is censored: if disliked tracks end more often with `endplay`, the skip rate is
  slightly biased.
- One listener: the result is about my listening, not about other users.

## Deviations
- 2026-10-27, clarification (no change to the design or the code): "same share of new tracks"
  in "Design" means the same limits on the share (10% to 40%), not the same share. In the
  planned weeks (notebook 05, built before any outcome) the MILP arm has 28.7% new tracks on
  average against 13.4% in the control arm (14% and 15% in the first week), because the model
  predicts low skip probabilities for new tracks. The secondary results for known and new tracks
  separately, already planned, separate the two effects.