"""Why the touch detector misses logged touches, and the ceiling of fixing each cause.

Every logged touch in live play (team_ceiling.logged_touches) with no detected touch within 8
frames gets one cause, the first that applies:
  unseen   the ball is not tracked within 8 frames of it
  reach4   a velocity peak is there, but the nearest player is beyond touch_reach (under 4 m)
  far      a velocity peak is there, nearest player 4 m or more away
  turn     a peak is there only without the 10 degree / 5 m/s change of flight filter
  weak     a peak is there only with touch_jerk lowered to 0.3
  nopeak   none of these
The ceiling of a cause: only those touches are added to the model's touch list with their true
team, and the model is run again. Games 1-2, seeds 0-2, clean leave-one-out dead ball classifier.
Game 3 and our match are not used. Test only.

    python notes/evidence/missed_touch_causes.py
"""
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import team_ceiling as T  # noqa: E402
from possession import validate as V, model  # noqa: E402

CAUSES = ("reach4", "unseen", "turn", "weak", "nopeak", "far")


def main():
    V.lock_holdout()
    p = dict(model.PARAMS)
    orig = model.detect_touches
    count = Counter()
    gains = {c: [] for c in CAUSES}
    for name in V.TUNE:
        true = T.logged_touches(name)
        for seed in V.SEEDS:
            t, labels, pred = T.E.predict(p, name, seed)
            s0 = T.score(labels, pred)[0]
            bx, by = model.fill_ball(t["bx"], t["by"], p["max_gap"])
            frames = orig(t, bx, by, p)[0]
            q = dict(p, touch_reach=1e9)
            anypk = orig(t, bx, by, q)[0]
            q = dict(q, touch_angle=0, touch_dspeed=0)
            rawpk = orig(t, bx, by, q)[0]
            lowpk = orig(t, bx, by, dict(q, touch_jerk=0.3))[0]
            _, _, dh, da = model.team_distances(t, bx, by)
            near = np.minimum(dh, da)
            by_cause = {c: [] for c in CAUSES}
            for f, tm, _ in true:
                if not 0 <= f < len(bx) or labels[f] not in "HA" or T.nearest(frames, f)[1] <= T.MATCH:
                    continue
                w = slice(max(0, f - T.MATCH), f + T.MATCH + 1)
                if not np.isfinite(t["bx"][w]).any() or not np.isfinite(bx[w]).any():
                    c = "unseen"
                elif T.nearest(anypk, f)[1] <= T.MATCH:
                    c = "reach4" if np.nanmin(near[w]) < 4 else "far"
                elif T.nearest(rawpk, f)[1] <= T.MATCH:
                    c = "turn"
                elif T.nearest(lowpk, f)[1] <= T.MATCH:
                    c = "weak"
                else:
                    c = "nopeak"
                count[c] += 1
                by_cause[c].append((f, tm, 0))
            for c in CAUSES:
                sub = np.array(sorted(by_cause[c]), dtype=np.int64).reshape(-1, 3)
                model.detect_touches = T.touch_fix(sub, "a")
                try:
                    gains[c].append(T.score(labels, T.E.predict(p, name, seed)[2])[0] - s0)
                finally:
                    model.detect_touches = orig
            print(name, seed, " ".join("%s %+.4f" % (c, gains[c][-1]) for c in CAUSES), flush=True)
    total = sum(count.values())
    print("\nmissed logged touches in live play: %d" % total)
    for c in CAUSES:
        print("%-7s %5d (%2.0f%%)  ceiling %+.4f  up %d/6" % (
            c, count[c], 100 * count[c] / total, np.mean(gains[c]), sum(g > 0 for g in gains[c])))


if __name__ == "__main__":
    main()
