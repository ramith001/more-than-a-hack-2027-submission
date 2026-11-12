"""Dead ball errors of the model on the bench (Games 1-2, seeds 0-2, clean leave-one-out inputs).

    python notes/evidence/dead_errors.py [name=value ...]

Per match and seed:
- false stoppages: predicted dead spells where play is live for most of the spell in the truth
  (count, median length, share with the ball missing for most of the spell)
- early starts: predicted dead spells starting over 1 s before the true stoppage they overlap
- late ends: predicted dead spells ending over 1 s after the true restart
- missed: true stoppages called live for most of their length
- goal kicks: logged goal kicks where the predicted stoppage runs on over 1 s past the kick
  (the kick's flight labelled dead), and how long it runs on
Test only; Game 3 is locked.
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "evaluation"))
import numpy as np  # noqa: E402
from possession import validate as V, model, restarts  # noqa: E402
from possession.external import events_for  # noqa: E402
from possession.io import PITCH  # noqa: E402

V.lock_holdout()
FPS = 25


def runs(mask):
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def inputs(name, seed, gap=False):
    """Tracks with p_dead from a classifier trained on the other tuning match, true labels, blocks."""
    pool = tuple(g for g in V.TUNE if g != name)
    params = dict(model.PARAMS, dead_gap_features=gap)
    V._clean_score(params, name, seed)
    return V._INPUTS[(name, seed, pool, True, gap)]


def predict(params, name, seed):
    t, labels, blocks = inputs(name, seed, params.get("dead_gap_features") or False)
    if params.get("use_restarts"):
        tags = restarts.simulated_tags(events_for(name), seed=seed)
        t = dict(t, restarts=restarts.restart_events(
            t, tags, pitch=PITCH if params.get("throw_ins") else None,
            moving=params.get("restart_moving", 6.0)))
    return t, labels, model.predict(t, PITCH, blocks, len(labels), params)


def errors(t, labels, pred):
    true_dead, pred_dead = labels == "D", pred == "D"
    graded = labels != "X"
    seen = np.isfinite(t["bx"])
    false, early, late = [], 0, 0
    true_runs = runs(true_dead)
    for a, b in runs(pred_dead & graded):
        if true_dead[a:b].mean() < 0.5:
            false.append((a, b, (b - a) / FPS, 1 - seen[a:b].mean()))
            continue
        over = [(c, d) for c, d in true_runs if c < b and d > a]
        if over and over[0][0] - a > FPS:
            early += 1
        if over and b - over[-1][1] > FPS:
            late += 1
    missed = sum(pred_dead[c:d].mean() < 0.5 for c, d in true_runs)
    return false, early, late, missed


def goal_kicks(name, labels, pred):
    """[(kick frame, seconds the predicted stoppage runs on past it)] for logged goal kicks that
    the truth calls live just after."""
    out = []
    for e in events_for(name):
        k = e["start"]
        if "GOAL KICK" not in e["subtype"] or k + FPS >= len(pred) or labels[k + 5] == "D":
            continue
        live = np.flatnonzero(pred[k:k + 30 * FPS] != "D")
        out.append((k, (live[0] if len(live) else 30 * FPS) / FPS))
    return out


def report(params):
    rows = []
    for name in V.TUNE:
        for seed in V.SEEDS:
            t, labels, pred = predict(params, name, seed)
            false, early, late, missed = errors(t, labels, pred)
            gk = goal_kicks(name, labels, pred)
            gk_late = [x for _, x in gk if x > 1.0]
            lens = [f[2] for f in false]
            missing = np.mean([f[3] > 0.5 for f in false]) if false else 0.0
            rows.append((len(false), early, late, missed, len(gk_late), len(gk)))
            print("{} seed {}  false {:3d} (median {:4.1f} s, ball missing {:.0%})  early {:3d}  late {:3d}"
                  "  missed {:3d}  goal kicks dead in flight {:2d}/{:2d} (median {:4.1f} s)".format(
                      name, seed, len(false), np.median(lens) if lens else 0, missing, early, late, missed,
                      len(gk_late), len(gk), np.median(gk_late) if gk_late else 0), flush=True)
    print("TOTAL false {}  early {}  late {}  missed {}  goal kicks dead in flight {} of {}".format(
        *np.sum(rows, axis=0)))


if __name__ == "__main__":
    params = dict(model.PARAMS)
    for arg in sys.argv[1:]:
        k, v = arg.split("=")
        params[k] = float(v) if "." in v else int(v)
    report(params)
