"""Where the team error is: the spellF1 ceiling of fixing one kind of team error perfectly.

Takes the model on the bench (Games 1-2, seeds 0-2, clean leave-one-out dead ball classifier,
current defaults) and replaces one thing with the true value from the Metrica events and labels:

  a  missed touches: every logged touch with no detected touch within 8 frames is added to the
     model's touch list with its true team, and the model is run again
  b  contested touches: every detected touch with players of both teams within reach that is
     matched to a logged touch but credited to the wrong team gets the true team, model run again
     (b+ does the same for every matched touch, contested or not, for scale)
  c  late handovers: where the truth changes team in live play and the prediction makes the same
     change more than 1 s (and at most 10 s) later, the change is moved to the true frame
  d  ball in flight: on the frames between the start and end of every logged pass or lost ball,
     a live frame given to the wrong team gets the true team
  e  short real spells: every true live spell shorter than min_spell (2 s) that the prediction
     gives mostly to the other team gets the true team

Logged touches: the start of every pass, recovery, lost ball, challenge, shot and set piece by its
From player, and the end of every completed pass by its To player. Detected touches are
model.detect_touches on the degraded tracking. a and b go through the model (touch evidence and
switch costs), so they are what a perfect touch detector would gain; c, d and e change the
output. Only graded live frames change. Game 3 and our match are not used. Test only.

    python notes/evidence/team_ceiling.py
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "notes", "evidence"))
import numpy as np  # noqa: E402

from possession import external as X, validate as V, model  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402
import dead_errors as E  # noqa: E402

FPS = 25
MATCH = 8          # frames between a logged and a detected touch to count as the same touch
TOUCH_TYPES = ("PASS", "RECOVERY", "BALL LOST", "CHALLENGE", "SHOT", "SET PIECE")


def slots(name):
    """{player name: track slot}, home slots first in file order, as external._load_csv_game."""
    folder = os.path.join(X.ensure_downloaded(), name)
    out, k = {}, 0
    for side in ("Home", "Away"):
        with open(os.path.join(folder, "{}_RawTrackingData_{}_Team.csv".format(name, side))) as f:
            header = [next(f) for _ in range(3)][2].strip().split(",")
        for h in header:
            if h.startswith("Player"):
                out[(side, h)] = k
                k += 1
    return out


def logged_touches(name):
    """Sorted array of (frame, team, slot) for every logged touch (see the module docstring)."""
    sl = slots(name)
    folder = os.path.join(X.ensure_downloaded(), name)
    out = []
    for e in csv.DictReader(open(os.path.join(folder, name + "_RawEventsData.csv"))):
        if e["Type"] not in TOUCH_TYPES or e["Team"] not in ("Home", "Away"):
            continue
        team = 0 if e["Team"] == "Home" else 1
        out.append((int(e["Start Frame"]) - 1, team, sl.get((e["Team"], e["From"]), -1)))
        if e["Type"] == "PASS" and e["To"]:
            out.append((int(e["End Frame"]) - 1, team, sl.get((e["Team"], e["To"]), -1)))
    return np.array(sorted(out), dtype=np.int64).reshape(-1, 3)


def nearest(frames, f):
    """Index of the frame in sorted `frames` nearest to f, and the distance."""
    i = np.searchsorted(frames, f)
    best = min((j for j in (i - 1, i) if 0 <= j < len(frames)), key=lambda j: abs(frames[j] - f),
               default=None)
    return (best, abs(int(frames[best]) - f)) if best is not None else (None, 10 ** 9)


def touch_fix(true, mode):
    """A wrapper for model.detect_touches that applies the truth: mode "a" adds missed touches,
    "b" corrects the team of contested matched touches, "b+" of every matched touch."""
    orig = model.detect_touches

    def detect(t, bx, by, p):
        frames, teams, contested = orig(t, bx, by, p)
        frames, teams, contested = frames.copy(), teams.copy(), contested.copy()
        if mode == "a":
            add = [(f, tm) for f, tm, _ in true if nearest(frames, f)[1] > MATCH and 0 <= f < len(bx)]
            if add:
                af, at = np.array(add).T
                frames = np.r_[frames, af]
                teams = np.r_[teams, at.astype(np.int8)]
                contested = np.r_[contested, np.zeros(len(af), bool)]
                o = np.argsort(frames, kind="stable")
                frames, teams, contested = frames[o], teams[o], contested[o]
        else:
            tf = true[:, 0]
            for i, f in enumerate(frames):
                j, dist = nearest(tf, f)
                if dist <= MATCH and (contested[i] or mode == "b+"):
                    teams[i] = true[j, 1]
        return frames, teams, contested
    return detect


def runs_of(x):
    ch = np.flatnonzero(np.r_[True, x[1:] != x[:-1]])
    return list(zip(ch, np.r_[ch[1:], len(x)]))


def fix_late(labels, pred, late=FPS, most=10 * FPS):
    new = pred.copy()
    for a, b in runs_of(labels)[1:]:
        before, now = labels[a - 1], labels[a]
        if before not in "HA" or now not in "HA" or before == now:
            continue
        # the first predicted change into `now` at or after a
        j = a
        while j < min(len(pred), a + most) and pred[j] != now:
            j += 1
        if j - a > late and j < a + most and pred[a] == before:
            seg = slice(a, j)
            new[seg] = np.where(np.isin(labels[seg], ["H", "A"]) & np.isin(pred[seg], ["H", "A"]),
                                labels[seg], new[seg])
    return new


def fix_flight(labels, pred, name):
    new = pred.copy()
    for e in X.events_for(name):
        if e["type"] not in ("PASS", "BALL LOST") or e["end"] <= e["start"]:
            continue
        seg = slice(e["start"], e["end"] + 1)
        live = np.isin(labels[seg], ["H", "A"]) & np.isin(pred[seg], ["H", "A"])
        new[seg] = np.where(live, labels[seg], new[seg])
    return new


def fix_short(labels, pred, m=None):
    m = m or int(model.PARAMS["min_spell"])
    new = pred.copy()
    for a, b in runs_of(labels):
        if labels[a] in "HA" and b - a < m and (pred[a:b] == labels[a]).mean() < 0.5:
            seg = slice(a, b)
            new[seg] = np.where(np.isin(pred[seg], ["H", "A"]), labels[a], new[seg])
    return new


def score(labels, pred):
    r = evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def main():
    V.lock_holdout()
    params = dict(model.PARAMS)
    keys = ("a", "b", "b+", "c", "d", "e")
    gains = {k: [] for k in keys}
    mgains = {k: [] for k in keys}
    changed = {k: 0 for k in keys}
    orig = model.detect_touches
    for name in V.TUNE:
        true = logged_touches(name)
        for seed in V.SEEDS:
            t, labels, pred = E.predict(params, name, seed)
            s0, m0 = score(labels, pred)
            out = {}
            for k in ("a", "b", "b+"):
                model.detect_touches = touch_fix(true, k)
                try:
                    out[k] = E.predict(params, name, seed)[2]
                finally:
                    model.detect_touches = orig
            out["c"] = fix_late(labels, pred)
            out["d"] = fix_flight(labels, pred, name)
            out["e"] = fix_short(labels, pred)
            row = []
            for k in keys:
                s1, m1 = score(labels, out[k])
                gains[k].append(s1 - s0)
                mgains[k].append(m1 - m0)
                changed[k] += int((out[k] != pred).sum())
                row.append("{} {:+.4f}".format(k, s1 - s0))
            print("{} seed {}  model {:.4f}  ".format(name, seed, s0) + "  ".join(row), flush=True)
    print()
    print("fix  mean spellF1 gain  runs up  mean macroF1 gain  frames changed per run")
    for k in sorted(keys, key=lambda k: -np.mean(gains[k])):
        print("{:3s}  {:+.4f}  {}/6  {:+.4f}  {:.0f}".format(
            k, np.mean(gains[k]), sum(g > 0 for g in gains[k]), np.mean(mgains[k]), changed[k] / 6))


if __name__ == "__main__":
    main()
