"""Where the dead ball error is: the spellF1 ceiling of fixing one kind of error perfectly.

Takes the model's predictions on the bench (Games 1-2, seeds 0-2, clean leave-one-out dead ball
classifier, current defaults) and replaces one thing with the true value from the Metrica labels:

  a  start of every stoppage: each predicted stoppage that overlaps a true one starts when it does
  b  end of every stoppage: each predicted stoppage that overlaps a true one ends when it does
  c  team after each restart: the first predicted live spell after each matched stoppage gets the
     team that really has the ball at the restart
  d  missed stoppages: true stoppages the model calls live for most of their length become dead
  e  invented stoppages: predicted stoppages that are mostly live in the truth become live

A predicted stoppage and a true one are matched when they overlap. Frames given back to play in
a, b and e keep the team the model has next to them (before the spell for a, after it for b, the
first half before and the second half after for e), so team errors are not fixed by accident; e
is also given with the true team on those frames (e*, upper bound). ab is a and b together. Only
graded frames are changed. Game 3 and our match are not used. Test only.

    python notes/evidence/dead_ceiling.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "notes", "evidence"))
import numpy as np  # noqa: E402

from possession import validate as V, model  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402
import dead_errors as E  # noqa: E402

LIVE = ("H", "A")


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def team_at(pred, i, step):
    """The model's team at the nearest live frame from i in direction step, or None."""
    while 0 <= i < len(pred):
        if pred[i] in LIVE:
            return pred[i]
        if pred[i] == "D":
            return None
        i += step
    return None


def matches(labels, pred):
    """[(pred run, true run)] for every overlapping predicted and true stoppage, graded frames."""
    graded = labels != "X"
    true_runs = runs((labels == "D") & graded)
    out = []
    for a, b in runs((pred == "D") & graded):
        for c, d in true_runs:
            if c < b and d > a:
                out.append(((a, b), (c, d)))
    return out


def fill(new, labels, sl, team):
    seg = new[sl]
    keep = labels[sl] == "X"
    new[sl] = np.where(keep, seg, team)


def fix_start(labels, pred):
    new = pred.copy()
    pairs = matches(labels, pred)
    first_true = {}
    for p, tr in pairs:
        first_true.setdefault(p, tr)
    used = set()
    for (a, b), (c, d) in first_true.items():
        if (c, d) in used:
            continue
        used.add((c, d))
        if c < a:
            fill(new, labels, slice(c, a), "D")
        elif a < c:
            team = team_at(pred, a - 1, -1)
            if team:
                fill(new, labels, slice(a, c), team)
    return new


def fix_end(labels, pred):
    new = pred.copy()
    last_true = {}
    for p, tr in matches(labels, pred):
        last_true[p] = tr
    used = set()
    for (a, b), (c, d) in sorted(last_true.items(), reverse=True):
        if (c, d) in used:
            continue
        used.add((c, d))
        if b < d:
            fill(new, labels, slice(b, d), "D")
        elif d < b:
            team = team_at(pred, b, 1)
            if team:
                fill(new, labels, slice(d, b), team)
    return new


def fix_team(labels, pred):
    """The predicted live spell that starts where a matched stoppage ends takes the true team at
    the restart (the first live true frame after the true stoppage)."""
    new = pred.copy()
    last_true = {}
    for p, tr in matches(labels, pred):
        last_true[p] = tr
    for (a, b), (c, d) in last_true.items():
        if b >= len(pred) or pred[b] not in LIVE:
            continue
        j = d
        while j < len(labels) and labels[j] not in LIVE:
            j += 1
        if j >= len(labels) or j - d > 50:
            continue
        e = b
        while e < len(pred) and pred[e] == pred[b]:
            e += 1
        new[b:e] = labels[j]
    return new


def fix_missed(labels, pred):
    new = pred.copy()
    for c, d in runs((labels == "D") & (labels != "X")):
        if (pred[c:d] == "D").mean() < 0.5:
            fill(new, labels, slice(c, d), "D")
    return new


def fix_invented(labels, pred, true_team=False):
    new = pred.copy()
    for a, b in runs((pred == "D") & (labels != "X")):
        if (labels[a:b] == "D").mean() >= 0.5:
            continue
        if true_team:
            new[a:b] = np.where(np.isin(labels[a:b], LIVE), labels[a:b], new[a:b])
            continue
        before, after = team_at(pred, a - 1, -1), team_at(pred, b, 1)
        before, after = before or after, after or before
        if before:
            m = (a + b) // 2
            fill(new, labels, slice(a, m), before)
            fill(new, labels, slice(m, b), after)
    return new


FIXES = {
    "a": fix_start,
    "b": fix_end,
    "ab": lambda l, p: fix_end(l, fix_start(l, p)),
    "c": fix_team,
    "d": fix_missed,
    "e": fix_invented,
    "e*": lambda l, p: fix_invented(l, p, true_team=True),
}


def score(labels, pred):
    r = evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)
    return r["spell_f1_mean"], r["macro_f1"]


def main():
    V.lock_holdout()
    params = dict(model.PARAMS)
    gains = {k: [] for k in FIXES}
    mgains = {k: [] for k in FIXES}
    changed = {k: 0 for k in FIXES}
    for name in V.TUNE:
        for seed in V.SEEDS:
            _, labels, pred = E.predict(params, name, seed)
            s0, m0 = score(labels, pred)
            row = []
            for k, fix in FIXES.items():
                new = fix(labels, pred)
                s1, m1 = score(labels, new)
                gains[k].append(s1 - s0)
                mgains[k].append(m1 - m0)
                changed[k] += int((new != pred).sum())
                row.append("{} {:+.4f}".format(k, s1 - s0))
            print("{} seed {}  model {:.4f}  ".format(name, seed, s0) + "  ".join(row), flush=True)
    print()
    print("fix  mean spellF1 gain  runs up  mean macroF1 gain  frames changed per run")
    for k in sorted(FIXES, key=lambda k: -np.mean(gains[k])):
        print("{:3s}  {:+.4f}  {}/6  {:+.4f}  {:.0f}".format(
            k, np.mean(gains[k]), sum(g > 0 for g in gains[k]), np.mean(mgains[k]), changed[k] / 6))


if __name__ == "__main__":
    main()
