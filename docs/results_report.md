# Results report

This report collects every score and check we have for the possession model. It adds to section 5 and 6 of the technical document and changes nothing in the model. The real match labels are private, so every number below comes from open data.

## 1. What is being scored

The model labels every frame of the match as home (H), away (A), dead ball (D) or unlabelled (X). Entries are ranked on spellF1 over those labels. The organisers' published floor on the full match is 0.2433 (smooth50).

## 2. Validation bench (Metrica matches 1 and 2)

Labels come from event data, tracking is degraded to look like the challenge tracking, and all settings were tuned on these two matches only under a strict rule: a change is kept only if it gains in every one of six runs, with a mean gain of at least 0.005. The dead-ball classifier is trained on these two matches and never on the match being scored.

| Bench | Without team probability | With team probability (final model) | Runs that gain |
|---|---|---|---|
| Metrica matches 1 and 2, three degradations | 0.615 | 0.645 | 6 of 6 |

The numbers in sections 2 and 3 come from `notes/evidence/team_evidence_all_five.py` (results in the `.json` and `.log` beside it). That bench takes the stretches of missing ball from the clean data's own gaps and never reads BAR-ATH. `possession/validate.py` runs an earlier form of the bench that copies gap patterns from the BAR-ATH tracking, so its numbers are different.

Resampling possessions gives a standard deviation of about 0.016, so differences of a few thousandths are noise.

## 3. Second bench (seven DFL open matches)

No hand-set setting was tuned on it. The provider's possession labels change hands more often than the Metrica labels, so scores are lower.

| Control rule | Without team probability | With team probability (final model) | Runs that gain |
|---|---|---|---|
| 1.5 s | 0.514 | 0.564 | 21 of 21 |
| 2.5 s | 0.562 | 0.606 | 21 of 21 |

The team probability with its first set of features failed here at weights 0.25 and 0.5 (5 November). The version in the model, with five more groups of features, was written down as a proposal and scored in one run on 8 November; one team member agreed to the proposal text on 9 November, after the run. For each scored match its classifier was trained on the other external matches only, so no match was scored by a classifier that saw it. Its features were chosen on Metrica, so these DFL rows are the independent part of the test. The smallest single gain was +0.006, and the mean gains (+0.050 and +0.043) are far above the +0.005 bar. The final model trains it on all seven DFL matches and Metrica matches 1 and 2.

## 4. Untouched match (Metrica match 3)

Match 3 stayed unseen until the hand-set model was final and was scored once, on 6 November, before the team probability was added. It was not used again, so the team probability has no match 3 result. The score was 0.4533, against 0.4382 for matches 1 and 2 pooled under the same scoring, a difference of +0.0151. A model that had overfit the tuning matches would score lower on the new match, and it did not.

## 5. Robustness on a different data source (SkillCorner open data)

We ran the hand-set model (before the team probability was added) unchanged on three SkillCorner open matches. They have no possession labels, so this checks behaviour, not score. SkillCorner tracking runs at 10 fps and was resampled to 25 fps, and the ball is missing in 26% to 33% of frames, which is worse than our bench.

| Match | Home | Away | Dead | Live-frame agreement with SkillCorner | Mean possession (s) |
|---|---|---|---|---|---|
| 1874553 | 0.21 | 0.36 | 0.43 | 0.845 | 14.4 |
| 1886347 | 0.32 | 0.23 | 0.44 | 0.849 | 14.4 |
| 1899585 | 0.26 | 0.24 | 0.50 | 0.796 | 11.0 |

The output is stable on all three: no failures or missing-value problems, possession lengths in a plausible range, and dead-ball share between 43% and 50% with the learned classifier supplying most of it. Agreement with SkillCorner's own live-ball frames is 80% to 85%. The lowest agreement and shortest possessions come from the match with the most missing ball (33%), as expected.

## 6. Reproducibility

`scripts/reproduce.sh <data> <out.csv>` builds the environment, runs the model, validates the file and prints its SHA256. The final model gives `046398fafb81c8a752c9853a8848d997301679437d1817a172244c1a9b043386` on six runs by the three team members: three fresh clones on one Mac (one with the DFL caches built from scratch), two fresh clones on a second Mac, and a Docker (linux/amd64) run on a Windows laptop from a fresh clone of the public repository. A separate check confirmed that the wired-in team classifier is the tested one: its training rows equal those of the test script bit for bit on all nine training matches, and its evidence term is identical. The earlier hand-set model gave its own hash identically on four runs: Docker (linux/amd64) on the Windows laptop and on a Mac, and natively on that Mac (Apple M4).

## 7. Known weak points

See section 7 of the technical document. The largest is missed touches: with every touch found and credited correctly, the bench score would rise by up to about 0.06.
