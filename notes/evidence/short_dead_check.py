"""Strict check of one detector for invented stoppages, the top item of dead_ceiling.py.

The detector (short_dead_inside below) is kept here and not in the model: it did not pass. It runs
after model.predict through a wrapper, with settings short_dead_len (frames, 0 = off) and
short_dead_line (metres). Games 1-2, seeds 0-2, clean leave-one-out, team rule, 500 bootstrap
draws. Game 3 and our match are not used. Test only.

    python notes/evidence/short_dead_check.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import numpy as np  # noqa: E402

from possession import validate as V, model  # noqa: E402

GRID = {"short_dead_len": [0, 100, 125, 150], "short_dead_line": [2.0, 4.0]}


def short_dead_inside(t, pitch, labels, max_len, line, look=75):
    """Play stops when the ball crosses a line or the referee stops it, and a free kick takes
    several seconds to set up. So a dead spell between two live spells, shorter than `max_len`
    frames, where the ball is seen within `look` frames before it and after it and every seen
    position from the last one before to the first one after is more than `line` metres from every
    line, is a tracking gap, not a stoppage. With the ball unseen nobody knows when it changed
    hands: the first half goes to the team before, the second half to the team after."""
    length, width = pitch
    bx, by = t["bx"], t["by"]
    seen = np.isfinite(bx)
    labels = labels.copy()
    ch = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    en = np.r_[ch[1:], len(labels)]
    for i in range(1, len(ch) - 1):
        a, b = ch[i], en[i]
        if labels[a] != "D" or labels[a - 1] not in "HA" or labels[b] not in "HA" or b - a >= max_len:
            continue
        before = np.flatnonzero(seen[max(0, a - look):a])
        after = np.flatnonzero(seen[b:b + look])
        if not len(before) or not len(after):
            continue
        w = np.arange(max(0, a - look) + before[-1], b + after[0] + 1)
        w = w[seen[w]]
        if np.min(np.minimum.reduce([bx[w], length - bx[w], by[w], width - by[w]])) <= line:
            continue
        mid = (a + b) // 2
        labels[a:mid], labels[mid:b] = labels[a - 1], labels[b]
    return labels


_predict = model.predict


def predict(t, pitch, blocks, n, p=model.PARAMS):
    labels = _predict(t, pitch, blocks, n, p)
    if p.get("short_dead_len"):
        labels = short_dead_inside(t, pitch, labels, int(p["short_dead_len"]),
                                   float(p.get("short_dead_line", 2.0)))
    return labels


def main():
    V.lock_holdout()
    model.predict = predict
    base = dict(model.PARAMS, short_dead_len=0, short_dead_line=2.0)
    for p in V.grid(base, GRID):
        r = [V._clean_score(p, g, s) for g in V.TUNE for s in V.SEEDS]
        print("  %s mean %.4f  %s" % ({k: p[k] for k in GRID}, np.mean(r), " ".join("%.4f" % x for x in r)),
              flush=True)
    V.report_loo(base, GRID, base, 500)


if __name__ == "__main__":
    main()
