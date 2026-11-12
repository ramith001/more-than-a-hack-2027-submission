# Team guide

Our working notes for the hackathon. The organisers' docs (README, CHALLENGE, DATA, EVALUATION,
RULES) are still the rules; this file is just how we work.

## Setup

```bash
git clone https://github.com/ramith001/more-than-a-hack-2027-custom
cd more-than-a-hack-2027-custom
git checkout main
pip install -r requirements-starter.txt
```

### The data

Download the dataset from the Google Drive link. Drive splits it into several zips; unzip all of
them and move the contents into `data/` so it looks like this:

```
data/
├─ tactical/
│  ├─ BAR-ATH_tactical.mp4
│  ├─ BAR-ATH_tactical_atd.xml
│  ├─ BAR-ATH_tactical_atd.txt
│  ├─ BAR-ATH_tactical_homography.csv
│  └─ BAR-ATH_tactical_smart_tagging.xml
└─ broadcast/
   ├─ BAR-ATH_broadcast.mp4
   ├─ BAR-ATH_broadcast_atd.xml
   ├─ BAR-ATH_broadcast_atd.txt
   └─ BAR-ATH_broadcast_homography.csv
```

The videos download with names like `BAR-ATH_tactical-001.mp4`. Rename them to the names above or
the code will not find them. The videos are only needed for video work; everything else runs
without them.

**Never commit anything in `data/`.** The repo has to be public and the data cannot be
redistributed. `.gitignore` already excludes it, so do not force-add it.

## Running it

```bash
python possession/run.py                  # writes evaluation/results/v2.csv
python evaluation/validate_submission.py evaluation/results/v2.csv
python evaluation/run_evaluation.py       # scores everything in evaluation/results/ on the public 5 min
python possession/validate.py             # scores the model on the validation bench (use this to decide)
```

The first run parses the 200 MB tracking file and caches it as `data/tactical/cache_tracking.npz`
(about 15 seconds). After that the whole match runs in a few seconds.

### The validation bench

`possession/external.py` builds three full validation matches from Metrica's public sample data
(downloaded automatically into `data/external/` on first run, about 30 seconds):

- possession labels come from the event data: possession changes at the new team's first
  controlling event (pass, recovery, shot, set piece); the ball is dead from going out, a foul or
  a goal until the next set piece
- the tracking is damaged to look like ours: the ball is smoothed the way Metrica's automatic
  pipeline smooths ours, extra live-ball gaps are copied from our match, and player tracks get
  noise and dropouts. This is the bench `possession/validate.py` runs. The final numbers in the
  technical document come from a later form (`notes/evidence/team_evidence_all_five.py`) that
  takes the gaps from the clean data itself and never reads our match, so `validate.py` gives
  different numbers

Goal kicks: Metrica never logs a goal kick as a SET PIECE. It is the PASS, BALL LOST or BALL OUT
the kick turns into, with GOAL KICK in the subtype, so `external.is_restart` counts those as
restarts. Before that fix (30 Oct) the ball stayed dead after every goal kick until the next logged
set piece, labelling 11k to 15k frames of live play per match as dead (dead share 50/43/38%, now
40/35/27%), and the dead ball classifier learned from it. The label cache is `<game>_v4.npz` (`_v5` with `LABEL_FIXES` on).

Labels are a proxy for the real ground truth, so use the bench to compare versions, not to predict the final score.

Bench labels follow CHALLENGE.md and the event data only. An earlier rule that handed possession
to the winner of a tackle that knocks the ball loose was chosen by looking at the public window, so
it was removed (2 Nov): CHALLENGE.md says a loose or contested ball stays with the last team that
had it. Every adopted setting was rechecked without it (Games 1 and 2, seeds 0 to 2, dead ball
classifier never trained on Game 3): 3 s minimum spell +0.045 and keeping restart spells +0.008,
both up on every match and seed; switch costs 60/10 still beat 20/7 (+0.013); the touch settings
+0.039 on average but down on one of the six (Game 2 seed 2, -0.012).

### Reproducing the submission

One command regenerates the submission file from the match data:

```bash
scripts/reproduce.sh /path/to/data              # writes evaluation/results/aditya_reproduce.csv
scripts/reproduce.sh /path/to/data out.csv      # second argument changes the output file
```

`DATA_DIR` can also be set as an environment variable. The data directory needs `tactical/` as the
organisers provide it. The script does not write into `DATA_DIR`. Cache files in `DATA_DIR` are
ignored. Running it on a checkout that already has the data in `data/` works too
(`scripts/reproduce.sh data`).

The script:

- creates `.venv` with Python 3.13 (set `PYTHON` to use another binary, `VENV` for another folder;
  it stops if the venv has another Python version) and installs `requirements-model.txt` from wheels,
  where every package is pinned
- sets all thread counts to 1 and fixes the hash seed
- runs `possession/run.py` and `evaluation/validate_submission.py`, then prints the SHA256 of the output

The Metrica sample data is fetched at a pinned commit (`COMMIT` in `possession/external.py`). A clone
at another commit gives a warning and is used as it is.

The tracking cache `cache_tracking.npz` now stores a key (raw file size, mtime and `CACHE_VERSION` in
`possession/io.py`) and is rebuilt when the key changes. Bump `CACHE_VERSION` when the parser changes.

Expected result of `scripts/reproduce.sh` (or the Dockerfile) on a fresh clone of `main`
(dead-ball classifier trained under a spread of degradations, team classifier on Metrica 1-2 and the seven DFL matches, 9 Nov), on any machine. Only the four tactical text files are needed; the Metrica and DFL data are downloaded:

```
SHA256 046398fafb81c8a752c9853a8848d997301679437d1817a172244c1a9b043386
```

The `Dockerfile` at the repo root runs the same script in a pinned Linux image (base image pinned by
digest) for anyone who prefers a container:

```bash
docker build --platform linux/amd64 -t mtah-possession .
docker run --rm --platform linux/amd64 -v /path/to/data:/data:ro -v "$PWD/out":/out mtah-possession
```

Always run the model through `reproduce.sh` (or its `.venv`). The earlier hash `c3cfe2b2...59fa` came
from system packages instead of the pinned ones and is withdrawn.

The first run takes a few minutes; later runs take seconds, since the caches live in `data/tactical/`. The
hash changes whenever the model changes, so compare against the hash posted with each merge.

`requirements.txt` stays the scorer's minimal list. `requirements-model.txt` is only for running the model.

### Overfitting guard

Overfitting is the biggest danger in this challenge. The rules we follow:

- **Never tune on the public 5 minutes.** It is only for checking the pipeline works.
- **Tune only on the two tuning matches** (Games 1 and 2). Game 3 is **locked**: look at it only
  to check a version before submitting (`python possession/validate.py --holdout`), never to
  choose between settings.
- **Every tuning step gets a leave-one-match-out check** (`python possession/validate.py --cv`):
  settings are picked without one match and scored on it. If the gain disappears on the match
  left out, the change is overfit and does not go in.
- **Pick from plateaus, not peaks**, and check a gain holds across seeds (`score(seeds=(0, 1, 2))`).
- **Every setting must have a football reason** that goes in the technical document.
- **Label changes to the bench** (new label rules, not new settings) cannot be scored against the
  old labels. They pass when bench dead ball agrees at least as well with Metrica's clean ball
  missing, and every setting re-picked on the new labels passes the rule there (agreed 3 Nov).
  Fixes A and B passed.
- No exceptions are in force. The touch settings (2.5 m, 3 frames), kept as an exception on the 3 s
  bench, beat the old ones on all 12 runs (seeds 0-5) of the 1.5 s bench (3 Nov).

v1 passed the leave-one-out check: out-of-sample gain over v0 of +0.113, +0.108 and +0.102 on the
three matches, with almost identical settings chosen each time.

### About X

X on a graded frame can never score better than a real label. For spellF1 the scorer replaces
each X with our previous label, so it is the same as writing that label ourselves; for macroF1 it
counts as a miss. Write X only on frames that are not graded.

## Dead ball detection (`possession/deadball.py`)

Two small gradient-boosted classifiers trained on the validation matches give P(dead) per frame:
one from ball signals (speed, standing still, outside the lines, distance to the lines and to the
nearest player) plus player signals, used where the ball is seen; one from player signals alone
(speed and whether players are slowing down or speeding up), used where it is not.

**Where the ball goes missing.** The ball is missing on 41% of graded frames, and almost all of
that is dead ball time: gaps are never short (none between 1 and 5 s; a third are 5 to 20 s, two
thirds longer than 20 s, matching stoppage lengths), the broadcast tracking rarely has the ball
either, and players move exactly as they do in stoppages (median 1.05 m/s, against 1.21 in known
stoppages and 2.33 in known live play). So "ball seen" is a signal for the player model. The
public 5 minutes are unusual here: the ball stays tracked through most of their stoppages.
The number of tracked players differs between the datasets and is left out.

A stoppage lasts at least 3 seconds (97% of the stoppages in the validation matches do). Before
each half's kick-off the ball is forced dead, and for the first second after every kick-off it is
forced live (kick-offs come from Smart Tagging plus the ball leaving the centre spot).

**Breaks and the ends of halves (30 Oct).** Our match has two hydration breaks, both taken in a
goal stoppage: about 31' (dead 58080 to the kick-off at 62094) and about 74' (dead 157546 to the
kick-off at 163083). In both, players walk to the bench-side touchline at halfway and most of them
leave the tracked pitch. Ball dead is also forced wherever fewer than half of the 22 players are on
the tracked pitch over 10 s (`players_off` in io.py). Play cannot go on without them, and in live
play the count never gets near it. On our match the rule fires only in stoppages that were already
dead, except one false H possession after the first-half whistle (90222 to 90391), which is now D.
The bench never drops half the players at once and does not use forced intervals, so its scores do
not move. The forced dead ball before a kick-off now runs to the frame before the detected kick,
not 5 frames before. 113742 is the published second-half kick-off and is detected exactly, and the
4 frames before it were labelled H.

Checked and right: everything outside the halves is X, the first half is live from its first
graded frame, the second half is D from 110808 up to 113742, and the 74' break is D the whole way
through. Still wrong:
- In the 31' break, an 8.5 s H at 61549 to 61760. A ball is tracked moving slowly near one goal
  while all 22 players stand back on the pitch (mean speed 0.9 m/s).
- After both break kick-offs, 9 to 12 s of D while the ball is lost (62120 to 62333 and 163198 to
  163487). The tracking does not show these are wrong. After the 74' kick-off, which is seen from
  the spot, the ball is lost 4 s later, the players slow to stoppage speed (1.2 m/s) and the ball
  comes back on the touchline. The 31' kick-off is not seen from the ball, so its frame (62094) is
  timed from the tag. For 16 s after it the players' mean speed stays at 0.7 to 0.9 m/s; after every
  kick-off seen from the ball it is over 1.5 m/s within 4 s.

  `kickoff_seen_only` (off) forces the ball live only after kick-offs seen from the spot. It is
  bench-neutral: the bench has no forced kick-off intervals. On the current main it changes no
  frame of our match, so it stays off (4 Nov).
- After the final whistle, a 6 s A at 189920 to 190075, with the ball tracked and players walking
  (0.8 m/s).

**Restarts.** For set pieces Smart Tagging gives the team, so that team has the ball for the first
moment after the restart. Throw-ins are tagged without a team; for those the team is the one of
the player holding the ball at the touchline just before it is thrown (right 94 to 100% of the time
on the validation matches; the last touch before the ball went out was a coin flip).

## Football rule

Football rule (Ramith, 2 Nov): the ball is dead from the whistle until a player restarts play. No
possession is gained or lost in between, whoever carries the ball to the spot; possession starts
again at the restart, with the team the referee awarded it to. After an advantage that comes to
nothing, the stoppage starts at the whistle, not at the original foul.

## How the model fits together

```
possession/
├─ io.py      loading: tracking as numpy arrays in metres, graded frames, writing a submission
├─ model.py   the model
└─ run.py     produces the submission file
```

`model.py` works in three steps:

1. **Clean the ball.** Short gaps in the ball track are filled in.
2. **Per-frame evidence.** For every frame, a score (log-likelihood) for each of H, A and D.
3. **Segmentation.** A Viterbi pass picks the best H/A/D sequence, with a cost for every switch,
   so the output is whole possessions instead of flickering frames.

Everyone's work plugs into step 2 or step 3. A better dead ball signal is a better D column; a
better ball position or control rule is better H and A columns. Keep that interface the same and
our parts will merge cleanly. All tunable numbers live in `PARAMS` at the top of `model.py`.

## Who does what

| | owns | first job |
|---|---|---|
| **Ramith** | live possession: who has the ball in play, what counts as a real change of possession, loose balls and duels, team shape when the ball is missing. Also makes the football calls on possession rules. | control rules and handover timing |
| **Aditya** | validation pipeline and reproducibility of the final file | external validation set from Metrica's free sample matches |
| **Aslan** | dead ball and restarts: out of bounds, stoppages, set pieces, Smart Tagging | find out whether v0's 47% dead ball on the full match is right |

### The learned team probability

| | |
|---|---|
| **Ramith** | first version of the learned team evidence (`notes/evidence/team_evidence_prototype.py`); wiring into `possession/team_evidence.py`, `model.py` and `run.py`, and the check that the wired code equals the tested one |
| **Aslan** | wrote the five groups of team features themselves (`notes/evidence/team_evidence_features.py`, now `team_evidence.extra()`, `attack_sign()` and `trailing()` in the model), the tests of the single feature groups, the five-group test script (`notes/evidence/team_evidence_all_five.py`) and the draft of its proposal; on the dead-ball side, the goal kick label fix, the placed ball in stoppages, and the players-off and kick-off rules; recomputed the final results and reran the later checks on his machine |
| **Aditya** | ran the five-group test on Metrica and DFL, made the review fixes to the script and proposal, and ran the reproduction checks |

## How we score ourselves

The board ranks on **spellF1**: each possession is matched as a whole against the truth, and
timing matters (1 second late drops a perfect possession from 1.0 to 0.65).

We only have ground truth for the first 5 minutes (17 possessions). **Do not tune on it.** It is
only for checking the pipeline works. The public window is also not typical: the ball is visible
88% of the time there against 59% across the match. Decide whether a change helps using the
validation bench.

Current scores on the bench (realistic version). Every number is out-of-sample: the dead ball
classifier is never trained on the match it is scored on. Game 3 is the locked one.

| | Game 1 | Game 2 | Game 3 (locked) |
|---|---|---|---|
| 30 Oct (goal kick fix, controlled possession labels), an earlier bench | 0.658 | 0.590 | not looked at |

These are old. Final numbers, on the final bench (`notes/evidence/team_evidence_all_five.py`): Games 1 and 2 0.615 without the team probability and 0.645 with it; DFL 0.514 to 0.564 at 1.5 s and 0.562 to 0.606 at 2.5 s. Game 3 was scored once, on 6 November, on the model before the team probability (0.4533 against 0.4382 for Games 1 and 2 under that script's random degradations) and not again.

**Bench fix (30 Oct).** The wandering ball error added to the validation matches was rough, not
smooth: it gave 42 cm of frame-to-frame jitter against about 1 cm on our match, so the touch
detector fired about 120 times a minute on the bench against 16 on our match, almost all of it
noise. The error is now smoothed (jitter about 0.8 cm, 17 to 19 touches a minute, the same
distance from the ball to the nearest player at a touch as ours). Switch costs were briefly lowered after this fix, but under the stricter check below that gain
did not hold once the touch settings were fixed, so they are back at 60 and 10. The player tracks had the same problem (43 cm of frame-to-frame jitter against 0.14 cm on
ours) and are now also given a smooth error instead.

Everything rejected on the old bench was then retested. Two settings now clearly help: a touch
counts with a player within 2.5 m instead of 1.5 m (our tracked ball sits about 0.9 m off the
foot), and the player credited with a touch is the one closest in the 3 frames before it (the
smoothed ball shows a touch late). Leave-one-match-out gain +0.041, the same settings chosen
every time, on a plateau (2.5 to 3.0 m, 2 to 5 frames). Rechecked under a stricter rule (Game 3
kept out entirely, including from the dead ball classifier; settings picked on one tuning match
and scored on the other; seeds 0 to 2): gain over the previous settings on both matches and all
three seeds, mean +0.043. With these touch settings the switch costs are flat between 20 and 60
and switch_touch 10 beats 7, so the switch costs stay at 60 and 10. Still rejected on the fixed bench: the
"ball follows the dribbler" evidence (follow_weight, +0.006 on the tuning matches but no
out-of-sample gain), a touch with both teams in reach never handing over, the reception path
rule, confirming touches, control spells, forcing play live after restarts, and making a lost
ball less likely dead when players run.

**Controlled possession (CHALLENGE.md).** Possession only changes hands when the new team has
controlled possession: a player keeps the ball for a few seconds, or the team completes more than
two consecutive passes. The bench labels follow this (`CONTROL_SECONDS` in external.py): a
spell won straight from the other team that lasts under the threshold with fewer than three passes goes back
to the team that had the ball. No provider publishes a number of seconds (Opta only says possession
ends when the opposition gains control). The bench uses 1.5 s, and the DFL bench is scored at 1.5 s and 2.5 s. The threshold was first
tested from 2 to 6 s, with the bench at 3 s: the model's matching rule (`min_spell`, 75 frames: a spell under
3 s goes to the team around it) gained on both tuning matches and all three seeds whenever the threshold was
3 s or more (+0.037 at 3 s, +0.04 to +0.06 at 4 to 6 s) and was neutral at 2 s. Those numbers were measured
at 3 s. On the 1.5 s bench with label fixes A and B, `min_spell` was re-picked under the strict rule:
50 frames (2 s) against 75, +0.025 out of sample, gaining on all six runs (3 Nov).
The touch settings and switch costs were rechecked on the new labels and still hold.

After the goal kick fix to the bench labels everything above was rechecked (strict rule, seeds 0 to 2):
the touch settings (+0.040) and the 3 s minimum spell (+0.038) still gain on both matches and all
seeds. touch_jerk 2.0 to 3.0 gains +0.01 on average but loses on Game 1 seed 0 at every value, so it
stays at 1.5 for now.

**Placed ball in stoppages (30 Oct).** The validation matches had almost no stoppages with the
ball still tracked (1 to 16% of dead frames), while our match has them: in 26% of set piece waits
the ball is seen for over 80% of the wait, at a median 1.65 m/s and 3.6 m from the restart spot
(it is brought to the spot and waits there). The dead ball classifier therefore learned that a
seen ball means live play. `external.WAITING_SHARE = 0.25` now fills the missing ball through a
random quarter of the bench's stoppages, starting a few metres off and settling onto the restart
spot (`_waiting_ball`, its own random stream, so the rest of the damage is unchanged). The bench
then has the ball seen in the last second before 25 to 31% of restarts (ours: 31%).

Strict check (Game 3 kept out, classifier for one tuning match trained on the other, seeds 0 to
2), everything scored on the 25% bench, only the training share changing: training at 0.15 /
0.25 / 0.35 gains +0.013 / +0.009 / +0.007 over 0, a plateau. Settings picked on one match and
scored on the other gain +0.019 (Game 1) and +0.008 (Game 2). The gain holds if our real share is
lower or higher (+0.008 on a 15% bench, +0.020 on a 35% bench), and costs nothing measurable if
it were 0 (-0.002, mixed). 0.25 is the share measured on our match, not the peak.

New bench, `validate.py`, seeds 0 to 2: Game 1 0.664 / 0.641 / 0.657, Game 2 0.590 / 0.639 /
0.608, mean spellF1 0.633, macroF1 0.823 (not comparable with the scores above: the bench
changed).

On our match the final model gives 249 possessions.

Published floor on the full match: spellF1 **0.2433**.

## Working together

- Make your own branch off `main`, e.g. `aditya/ball-detection`.
- Commit small and often, push your branch, and do not push straight to `main`.
- Before merging, run `python possession/run.py` and the scorer and post the numbers in the gc.

## Rules we cannot break

- Labels must come from the algorithm. No hand labelling or hand correcting anything on this match.
- Organisers' rules (email, 4 Nov): no checking the output against the video; no hand labels or
  other manually created data on the provided game; no training, fitting or calibrating on the
  ground truth or the provided video. Every trained or tuned value comes from external data (the
  Metrica bench). Allowed: automatic checks using only the provided data, inspecting the data and
  the video to understand what counts as possession, the video and generated data as input,
  intermediate files, and external datasets we have the rights to.
- Never use the private ground truth or any other possession data for this fixture.
- The repo must regenerate the exact submission file from the data, so no manual steps.
- Anything we use must be accessible to the organisers: no private models or private data.

## Timeline

Deadline **November 17, 2026**.

- validation set working, each part has a first version
- all parts merged into one model
- improving with the validation set
- freeze, technical document, submit
