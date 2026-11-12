"""Prototype: learned team evidence (who has the ball in live play), test only, nothing wired in.

A gradient boosted classifier predicts P(home has the ball) per frame from ball and player features
(distance to the nearest players of each team, speeds, time since each team last touched the ball, last
touch owner). It is trained on the other tuning match only (Metrica Game 1 or 2, never Game 3, never
the challenge match) and scored on the match left out, three degradation seeds, dead ball classifier
also trained on the other match. Modes: "learned" replaces the proximity, touch and follow evidence,
"added" adds it to them with a weight. Run from the repo root:
    python notes/evidence/team_evidence_prototype.py added 0.5
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "evaluation"))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from possession import external as X, model, deadball, validate as V
from possession.external import PITCH
from possession_metrics import evaluate, label_codes
V.lock_holdout()
P = dict(model.PARAMS)

def feats(t, p=P):
    bx, by = model.fill_ball(t["bx"], t["by"], p["max_gap"])
    n = len(bx); ok = np.isfinite(bx)
    d, tf, dh, da = model.team_distances(t, bx, by)
    speed, jerk = model.ball_kinematics(bx, by)
    f = [dh, da, np.minimum(dh, da), da - dh, speed, jerk, ok.astype(float)]
    for team in (0, 1):
        dd = np.where(tf == team, d, np.inf)
        s = np.sort(dd, axis=1)
        f += [s[:, 1], (dd < 3).sum(1), (dd < 6).sum(1)]
        j = np.argmin(dd, axis=1)
        sp = np.hypot(np.gradient(np.where(np.isfinite(t["px"]), t["px"], np.nan), axis=0),
                      np.gradient(np.where(np.isfinite(t["py"]), t["py"], np.nan), axis=0))[np.arange(n), j] * 25
        f.append(sp)
    frames, teams, _ = model.detect_touches(t, bx, by, p)
    for team in (0, 1):
        last = np.full(n, 10 ** 4.0); fr = frames[teams == team]
        idx = np.searchsorted(fr, np.arange(n), side="right") - 1
        ok2 = idx >= 0
        last[ok2] = np.arange(n)[ok2] - fr[idx[ok2]]
        f.append(np.minimum(last, 750))
    owner = model.touch_owner(frames, teams, n, p)
    f.append(owner.astype(float))
    return np.column_stack(f).astype(np.float32), ok

def fit(name, seed):
    t, y, blocks = X.load_game(name, seed=seed)
    F, ok = feats(t)
    m = np.isin(y, ["H", "A"]) & ok
    idx = np.flatnonzero(m)[::5]
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
    clf.fit(F[idx], (np.asarray(y)[idx] == "H").astype(int))
    return clf

def make_emissions(mode, nw):
    orig = model.emissions
    def emis(t, pitch, p=P):
        ll = orig(t, pitch, dict(p, near_weight=0.0, touch_weight=0.0, follow_weight=0.0) if mode == "learned" else p)
        ph = t["p_team"]; ok = t["p_ok"]
        ph = np.where(ok, ph, 0.5)
        ll[:, 0] += nw * np.log(np.clip(ph, 1e-3, 1)); ll[:, 1] += nw * np.log(np.clip(1 - ph, 1e-3, 1))
        return ll
    return emis, orig

def run(mode, nw):
    emis, orig = make_emissions(mode, nw)
    out = []
    for name in V.TUNE:
        other = [g for g in V.TUNE if g != name][0]
        for seed in (0, 1, 2):
            t, y, blocks = X.load_game(name, seed=seed)
            t = dict(t, p_dead=deadball.p_dead(deadball.train([other], seed=seed), t, PITCH))
            F, ok = feats(t)
            clf = fit(other, seed)
            t["p_team"] = clf.predict_proba(F)[:, 1]; t["p_ok"] = ok
            base = model.predict(t, PITCH, blocks, len(y), P)
            model.emissions = emis
            try: new = model.predict(t, PITCH, blocks, len(y), P)
            finally: model.emissions = orig
            sc = lambda a: evaluate(label_codes(y), label_codes(a), bootstrap_n=0)["spell_f1_mean"]
            out.append((sc(base), sc(new)))
            print(mode, nw, name, seed, "%.4f -> %.4f" % out[-1], flush=True)
    g = [b - a for a, b in out]
    print("RESULT %s nw=%s mean gain %+.4f runs up %d/6 %s" % (mode, nw, np.mean(g), sum(x > 0 for x in g), V.verdict(g)), flush=True)

if __name__ == "__main__":
    run(sys.argv[1], float(sys.argv[2]))
