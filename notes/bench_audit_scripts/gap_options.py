"""Removing the challenge-match read from external.degrade: gaps removed or built from external data. Test only.

    MTAH_DFL_DIR=/path OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 .venv/bin/python notes/bench_audit_scripts/gap_options.py check
    ... gap_options.py matrix OUT.json        # classifier: training share x scoring share, current PARAMS
    ... gap_options.py sweep OUT.json SHARE   # PARAMS: every dfl_recheck single change and the near-miss combinations

The challenge match is never read: no_challenge_read stubs the read, so every run starts from "no gaps".
Replacement ("own gaps"): the match's own clean-ball missing stretches on graded frames, after the ball
fill, moved by a random offset, each kept with probability SHARE. In Metrica those are its stoppages and
a few short live gaps; in DFL its stoppages, because the loader blanks the ball on BallStatus=0. Nothing
crosses from a scored match into the classifier that scores it. Own random stream [seed, 3], so the rest of
the damage is the same at every share and SHARE 0 is exactly the bench of no_challenge_read.

Runs: 7 DFL matches x seeds 0-2 at control 1.5 s and 2.5 s, classifier on Metrica Games 1-2; Metrica Games
1-2 x seeds 0-2, classifier on the other game. Metrica Game 3 is locked. Nothing here changes the model.
"""
import no_challenge_read as N      # audit guard first
import json
import multiprocessing as mp
import sys
import time

import dfl_recheck as R
from sklearn.metrics import log_loss, roc_auc_score
from possession_metrics import evaluate, label_codes
from possession import deadball, external, external_dfl, model, restarts, validate as V

np = R.np
PITCH = external.PITCH
SEEDS = (0, 1, 2)
DFL = external_dfl.GAMES
METRICA = V.TUNE
SHARES = (0.0, 0.1, 0.2, 0.3)
SHUF = ("seen12", "seen50", "seen125")
WORKERS = 5
EXTRA = [("combo", "switch_control={}+gap={}".format(sc, g), dict(model.PARAMS, switch_control=sc, dead_gap_features=g))
         for sc in (5, 10) for g in ("ball", "all")]

N.install()
V.lock_holdout()
_REAL_DEGRADE = external.degrade
_SHARE = 0.0
_MODELS = {}
_EVENTS = {}


def own_gap_mask(t, labels, seed, share, fill_frames):
    bx = t["bx"]
    if fill_frames:
        bx, _ = external._fill_short(t["bx"], t["by"], labels, fill_frames)
    rng = np.random.default_rng([seed, 3])
    missing = ~np.isfinite(bx)[labels != "X"]
    mask = np.resize(np.roll(missing, rng.integers(0, len(missing))), len(bx))
    out = np.zeros(len(bx), dtype=bool)
    for a, b in zip(*external._runs(mask)):
        if rng.random() < share:
            out[a:b] = True
    return out


def degrade(t, seed=0, labels=None, fill_frames=0, **kw):
    out = _REAL_DEGRADE(t, seed=seed, labels=labels, fill_frames=fill_frames, **kw)
    if _SHARE:
        m = own_gap_mask(t, labels, seed, _SHARE, fill_frames)
        out["bx"], out["by"] = np.where(m, np.nan, out["bx"]), np.where(m, np.nan, out["by"])
    return out


external.degrade = degrade


def set_share(s):
    global _SHARE
    _SHARE = s


def pool_of(name):
    return tuple(g for g in METRICA if g != name)


def events(name):
    if name not in _EVENTS:
        _EVENTS[name] = (external_dfl.events_for(name) if name in DFL else external.events_for(name))
    return [dict(e) for e in _EVENTS[name]]


def load(name, seed, share):
    """Tracks, {label version: labels}, blocks. DFL tracks are the same at both control lengths (asserted in check)."""
    set_share(share)
    if name in DFL:
        t, lab, blocks = external_dfl.load_game(name, seed=seed, control=1.5)
        two = external.require_control(external_dfl.raw_labels(name), events(name), 2.5)
        return t, {"1.5": lab, "2.5": two}, blocks
    t, lab, blocks = external.load_game(name, seed=seed)
    return t, {"metrica": lab}, blocks


def _train(key):
    share, pool, seed, gap = key
    set_share(share)
    return key, deadball.train(list(pool), seed=seed, gap=gap)


def train_all(keys):
    todo = sorted(set(keys) - set(_MODELS), key=str)
    t0 = time.time()
    with mp.get_context("fork").Pool(WORKERS) as p:
        for key, m in p.imap_unordered(_train, todo):
            _MODELS[key] = m
    print("{} classifier pairs trained in {:.0f}s".format(len(todo), time.time() - t0), flush=True)


def pools():
    return [METRICA, METRICA[1:], METRICA[:1]]


def model_for(share, name, seed, gap=False):
    pool = METRICA if name in DFL else pool_of(name)
    assert name not in pool
    return _MODELS[(share, pool, seed, gap)]


def score_all(t, labels, blocks, tags_rest, p_dead, params):
    t = dict(t, p_dead=p_dead) if params.get("learned_dead") else dict(t)
    if params.get("use_restarts"):
        t["restarts"] = tags_rest(params)
    pred = model.predict(t, PITCH, blocks, len(next(iter(labels.values()))), params)
    res = {}
    for v, lab in labels.items():
        r = evaluate(label_codes(lab), label_codes(pred), bootstrap_n=0)
        res[v] = {k: float(r[k]) for k in ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells",
                                           "alive_dead_f1", "team_acc_both_alive")}
    return res, pred


def rest_fn(name, t, seed):
    tags = restarts.simulated_tags(events(name), seed=seed)
    cache = {}

    def f(params):
        k = (bool(params.get("throw_ins")), params.get("restart_moving", 6.0))
        if k not in cache:
            cache[k] = restarts.restart_events(t, tags, pitch=PITCH if k[0] else None, moving=k[1])
        return cache[k]
    return f


def clf_metrics(models, t, lab, p, seed):
    g = lab != "X"
    dead, seen, q = (lab[g] == "D"), np.isfinite(t["bx"])[g], np.clip(p[g], 1e-6, 1 - 1e-6)
    lm, dm = ~dead & ~seen, dead & ~seen
    _, P, _ = deadball.features(t, PITCH, False)
    names = list(deadball._feature_table(t, PITCH, False)[1])
    Pg = P[g][::5]
    y = dead[::5]
    rng = np.random.default_rng([seed, 9])
    base = roc_auc_score(y, models[1].predict_proba(Pg)[:, 1])
    order = rng.permutation(len(Pg))
    S = Pg.copy()
    for c in SHUF:
        i = names.index(c)
        S[:, i] = Pg[order, i]
    return dict(
        auc=float(roc_auc_score(dead, q)),
        auc_missing=float(roc_auc_score(dead[~seen], q[~seen])) if lm.any() and dm.any() else None,
        log_loss=float(log_loss(dead, q)),
        live_missing_share=float((~seen)[~dead].mean()), dead_missing_share=float((~seen)[dead].mean()),
        live_missing_p=float(q[lm].mean()) if lm.any() else None,
        live_missing_called_dead=float((q[lm] > 0.5).mean()) if lm.any() else None,
        dead_missing_called_dead=float((q[dm] > 0.5).mean()) if dm.any() else None,
        live_called_dead=float((q[~dead] > 0.5).mean()), dead_called_dead=float((q[dead] > 0.5).mean()),
        seen_auc_drop=float(base - roc_auc_score(y, models[1].predict_proba(S)[:, 1])))


def matrix_job(args):
    name, seed = args
    out = []
    for score in SHARES:
        t, labels, blocks = load(name, seed, score)
        rest = rest_fn(name, t, seed)
        preds = {}
        for train in SHARES:
            m = model_for(train, name, seed)
            p = deadball.p_dead(m, t, PITCH)
            res, pred = score_all(t, labels, blocks, rest, p, model.PARAMS)
            preds[train] = pred
            out.append(dict(name=name, seed=seed, train=train, score=score, res=res,
                            clf=clf_metrics(m, t, next(iter(labels.values())), p, seed)))
        g = next(iter(labels.values())) != "X"
        for row in out[-len(SHARES):]:
            row["frames_moved_vs_train0"] = float((preds[row["train"]][g] != preds[0.0][g]).mean())
    return out


def sweep_job(args):
    name, seed, share, cands = args
    t, labels, blocks = load(name, seed, share)
    rest = rest_fn(name, t, seed)
    dead = {}

    def one(params):
        gap = params.get("dead_gap_features") or False
        if params.get("learned_dead") and gap not in dead:
            dead[gap] = deadball.p_dead(model_for(share, name, seed, gap), t, PITCH)
        return score_all(t, labels, blocks, rest, dead.get(gap), params)[0]

    res = {"ref": one(model.PARAMS)}
    for _, label, p in cands:
        res[label] = one(p)
    return dict(name=name, seed=seed, res=res)


def runs():
    return [(n, s) for n in DFL + METRICA for s in SEEDS]


def run_pool(fn, todo, path, wrap):
    t0 = time.time()
    rows = []
    with mp.get_context("fork").Pool(WORKERS) as p:
        for k, row in enumerate(p.imap_unordered(fn, todo)):
            rows.append(row)
            json.dump(wrap(rows), open(path, "w"))
            print("[{:5.0f}s] {}/{}".format(time.time() - t0, k + 1, len(todo)), flush=True)
    return rows


def matrix(path):
    train_all([(s, pl, seed, False) for s in SHARES for pl in pools() for seed in SEEDS])
    rows = run_pool(matrix_job, runs(), path, lambda r: dict(rows=[x for b in r for x in b]))
    json.dump(dict(rows=[x for b in rows for x in b]), open(path, "w"), indent=1)


def sweep(path, share):
    cands = R.candidates()[0] + EXTRA
    gaps = {R.gap_of(p) for _, _, p in cands} | {False}
    train_all([(share, pl, seed, g) for pl in pools() for seed in SEEDS for g in gaps])
    print("{} candidates".format(len(cands)), flush=True)
    meta = dict(share=share, candidates=[(g, l) for g, l, _ in cands])
    rows = run_pool(sweep_job, [(n, s, share, cands) for n, s in runs()], path, lambda r: dict(meta, rows=r))
    json.dump(dict(meta, rows=rows), open(path, "w"), indent=1)


def check():
    """The wrapper changes only the ball, share 0 is the stub bench, the mask is deterministic, DFL tracks do
    not depend on the control length, and one DFL and one Metrica reference match the existing code paths."""
    set_share(0.0)
    a, lab, _ = external_dfl.load_game("J03WMX", seed=1, control=1.5)
    a2, _, _ = external_dfl.load_game("J03WMX", seed=1, control=2.5)
    set_share(0.2)
    b, _, _ = external_dfl.load_game("J03WMX", seed=1, control=1.5)
    b2, _, _ = external_dfl.load_game("J03WMX", seed=1, control=1.5)
    for k in ("px", "py", "ps"):
        assert np.array_equal(a[k], b[k], equal_nan=True) and np.array_equal(a[k], a2[k], equal_nan=True), k
    assert np.array_equal(a["bx"], a2["bx"], equal_nan=True)
    assert np.array_equal(b["bx"], b2["bx"], equal_nan=True)
    ok = np.isfinite(b["bx"])
    assert np.array_equal(a["bx"][ok], b["bx"][ok]) and not (np.isfinite(a["bx"]) < ok).any()
    assert np.array_equal(np.isfinite(b["bx"]), np.isfinite(b["by"]))
    g, dead = lab != "X", lab == "D"
    print("J03WMX seed 1 ball missing, live: none {:.3f} own 0.2 {:.3f}; dead: none {:.3f} own 0.2 {:.3f}".format(
        (~np.isfinite(a["bx"]))[g & ~dead].mean(), (~ok)[g & ~dead].mean(),
        (~np.isfinite(a["bx"]))[dead].mean(), (~ok)[dead].mean()))
    train_all([(0.0, METRICA, 1, False), (0.0, METRICA[1:], 1, False)])
    set_share(0.0)
    mine = sweep_job(("J03WMX", 1, 0.0, []))["res"]["ref"]
    V._DEAD[(METRICA, 1, False)] = _MODELS[(0.0, METRICA, 1, False)]
    theirs = R.run_job(("J03WMX", 1, 1.5, [], None))["res"]["ref"]["spell_f1_mean"]
    print("J03WMX seed 1 control 1.5 mine {:.6f} dfl_recheck {:.6f}".format(mine["1.5"]["spell_f1_mean"], theirs))
    assert mine["1.5"]["spell_f1_mean"] == theirs
    m2 = sweep_job(("Sample_Game_1", 1, 0.0, []))["res"]["ref"]["metrica"]["spell_f1_mean"]
    V._DEAD[(METRICA[1:], 1, False)] = _MODELS[(0.0, METRICA[1:], 1, False)]
    t2 = V._clean_score(model.PARAMS, "Sample_Game_1", 1)
    print("Game 1 seed 1 mine {:.6f} validate {:.6f}".format(m2, t2))
    assert m2 == t2
    print("check passed")


if __name__ == "__main__":
    t_start = time.time()
    mode = sys.argv[1]
    if mode == "check":
        check()
    elif mode == "matrix":
        matrix(sys.argv[2])
    elif mode == "sweep":
        sweep(sys.argv[2], float(sys.argv[3]))
    print("wall time {:.0f}s".format(time.time() - t_start))
