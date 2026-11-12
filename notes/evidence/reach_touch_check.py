"""Strict check of one detector for missed touches, the top cause in team_ceiling.py.

Missed logged touches by cause (Games 1-2, seeds 0-2, 5446 in live play): a velocity peak with the
nearest player 2.5 to 4 m away 33%, ball unseen 25%, turn or speed change too small 23%, no peak
8%, peak too weak 6%, nearest player 4 m or more 6% (missed_touch_causes.py). Adding only the
first group with its true team gains +0.0274 (6/6 runs), the unseen ball +0.0190 (6/6), the others
under +0.01.

The detector: the tracking puts players about a metre off where their foot meets the ball, so a
velocity peak with the nearest player beyond `touch_reach` but within `reach_far` metres is also a
touch, by that player's team, when the other team's nearest player is at least `reach_margin`
metres further away (only one team could have touched it). It runs as a wrapper around
model.detect_touches, so it feeds the touch evidence and switch costs. Settings reach_far (metres,
0 = off) and reach_margin. Games 1-2, seeds 0-2, clean leave-one-out, team rule, 500 bootstrap
draws. Game 3 and our match are not used. Test only.

    python notes/evidence/reach_touch_check.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import numpy as np  # noqa: E402
from scipy.ndimage import minimum_filter1d  # noqa: E402

from possession import validate as V, model  # noqa: E402

GRID = {"reach_far": [0, 3.0, 3.5, 4.0], "reach_margin": [1.0, 2.0]}
_detect = model.detect_touches


def detect_touches(t, bx, by, p):
    frames, teams, contested = _detect(t, bx, by, p)
    far = p.get("reach_far") or 0
    if far <= p["touch_reach"]:
        return frames, teams, contested
    f2, t2, _ = _detect(t, bx, by, dict(p, touch_reach=far))
    _, _, dh, da = model.team_distances(t, bx, by)
    lag = int(p.get("touch_lag", 0))
    if lag:
        dh = minimum_filter1d(dh, lag + 1, origin=lag // 2)
        da = minimum_filter1d(da, lag + 1, origin=lag // 2)
    new = ~np.isin(f2, frames)
    near, other = np.minimum(dh[f2], da[f2]), np.maximum(dh[f2], da[f2])
    keep = new & (near > p["touch_reach"]) & (other >= near + p.get("reach_margin", 1.0))
    if not keep.any():
        return frames, teams, contested
    owner = np.where(dh[f2] <= da[f2], model.H, model.A).astype(np.int8)
    frames = np.r_[frames, f2[keep]]
    teams = np.r_[teams, owner[keep]]
    contested = np.r_[contested, np.zeros(keep.sum(), bool)]
    o = np.argsort(frames, kind="stable")
    return frames[o], teams[o], contested[o]


def main():
    V.lock_holdout()
    model.detect_touches = detect_touches
    base = dict(model.PARAMS, reach_far=0, reach_margin=1.0)
    for p in V.grid(base, GRID):
        r = [V._clean_score(p, g, s) for g in V.TUNE for s in V.SEEDS]
        print("  %s mean %.4f  %s" % ({k: p[k] for k in GRID}, np.mean(r), " ".join("%.4f" % x for x in r)),
              flush=True)
    V.report_loo(base, GRID, base, 500)


if __name__ == "__main__":
    main()
