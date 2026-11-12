"""Test set for the rule "a stoppage lasts until play is restarted" (for Aslan).

Football rule (Ramith, 2 Nov): the ball is dead from the whistle until a player restarts play
(free kick, throw-in, goal kick, corner, drop ball). Nobody gains or loses possession in between,
however the ball is carried about. On our match the model sometimes ends a stoppage early and
gives a team a short live spell while the ball is carried to the spot.

This script checks a restart detector and the rule on the Metrica validation matches only
(Games 1-2, seeds 0-2; Game 3 locked; nothing from our match's ground truth):

1. Detection: for every logged restart (set piece or goal kick), is a restart detected within
   1 s / 2 s of it, how many detections are not near a logged restart, and the timing error.
2. Bench effect: the model as it is against the model with the rule applied, clean
   leave-one-out classifier, spellF1 per match and seed, with the team verdict.

Write a detector with the signature detect(t, pred) -> sorted list of frames and pass it to
main(). The one below is the starting point: after each stoppage the model predicts, the first
moment the seen ball goes from still to fast (restarts.find_restart, the same test the tagged
restarts use).

    python notes/evidence/restart_rule_check.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import numpy as np  # noqa: E402

from possession import external as X, model, restarts, validate as V  # noqa: E402
from possession.external import PITCH  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402

FPS = 25
MAX_WAIT = 60 * FPS   # a stoppage is extended at most this far


def runs(mask):
    e = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)))


def detect_still_to_fast(t, pred):
    """After each predicted stoppage start, the first frame the seen ball is still and then fast."""
    out = []
    for a, _ in runs(pred == "D"):
        r = restarts.find_restart(t, a, window=MAX_WAIT)
        if r is not None:
            out.append(r)
    return sorted(set(out))


def apply_rule(pred, found):
    """Each predicted stoppage lasts until the first detected restart after its start (at most
    MAX_WAIT): live frames in between become dead."""
    pred = pred.copy()
    found = np.asarray(found)
    for a, b in runs(pred == "D"):
        nxt = found[(found > a) & (found < a + MAX_WAIT)]
        if len(nxt) and nxt[0] > b:
            seg = slice(b, nxt[0])
            pred[seg] = np.where(np.isin(pred[seg], ["H", "A"]), "D", pred[seg])
    return pred


def logged_restarts(name):
    ev = X.events_for(name)
    return np.array(sorted(e["start"] for e in ev if X.is_restart(e) and "KICK OFF" not in e["subtype"]))


def predictions(name, seed, params):
    """Model prediction on the bench with the clean leave-one-out dead ball classifier."""
    from possession import deadball
    t, labels, blocks = X.load_game(name, seed=seed)
    pool = [g for g in V.TUNE if g != name]
    t = dict(t, p_dead=deadball.p_dead(deadball.train(pool, seed=seed), t, PITCH))
    if params.get("use_restarts"):
        tags = restarts.simulated_tags(X.events_for(name), seed=seed)
        t = dict(t, restarts=restarts.restart_events(
            t, tags, pitch=PITCH if params.get("throw_ins") else None,
            moving=params.get("restart_moving", 6.0)))
    return t, labels, model.predict(t, PITCH, blocks, len(labels), params)


def main(detect=detect_still_to_fast, params=None, seeds=(0, 1, 2)):
    params = params or model.PARAMS
    V.lock_holdout()
    gains, near1, near2, errs, n_true, n_found, n_false = [], 0, 0, [], 0, 0, 0
    for name in V.TUNE:
        true = logged_restarts(name)
        for seed in seeds:
            t, labels, pred = predictions(name, seed, params)
            found = np.asarray(detect(t, pred))
            n_true += len(true)
            n_found += len(found)
            for r in true:
                d = np.abs(found - r) if len(found) else np.array([10 ** 9])
                near1 += d.min() <= FPS
                near2 += d.min() <= 2 * FPS
                if d.min() <= 2 * FPS:
                    errs.append((found[np.argmin(d)] - r) / FPS)
            n_false += sum(np.abs(true - f).min() > 2 * FPS for f in found) if len(true) else len(found)
            new = apply_rule(pred, found)
            s0 = evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)["spell_f1_mean"]
            s1 = evaluate(label_codes(labels), label_codes(new), bootstrap_n=0)["spell_f1_mean"]
            gains.append(s1 - s0)
            print("{} seed {}  model {:.4f}  with rule {:.4f}  gain {:+.4f}  frames changed {}".format(
                name, seed, s0, s1, s1 - s0, int((new != pred).sum())), flush=True)
    print("logged restarts {}  detected within 1 s {:.0%}, within 2 s {:.0%}  detections {}  "
          "not near a logged restart {}  median timing error {:+.2f} s".format(
              n_true, near1 / n_true, near2 / n_true, n_found, n_false,
              float(np.median(errs)) if errs else float("nan")))
    ok, why = V.verdict(gains)
    print("mean gain {:+.4f}  {}".format(np.mean(gains), "VERDICT PASS" if ok else "VERDICT FAIL: " + why))


if __name__ == "__main__":
    main()
