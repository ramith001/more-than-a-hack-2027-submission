"""Dead ball classifier trained under a spread of degradations (deadball.train_spread) against option C.

    MTAH_DFL_DIR=/path OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 .venv/bin/python notes/bench_audit_scripts/spread_training.py check
    ... spread_training.py bench OUT.json
    ... spread_training.py bench OUT.json MATCH SEED     # one job, for a determinism check
    .venv/bin/python notes/bench_audit_scripts/spread_training.py count CURRENT.csv NEW.csv

Scored tracks are the option C bench of gap_options (own gaps at share 0.2), the same for both classifiers.
DFL: 7 matches x seeds 0-2 at control 1.5 s and 2.5 s, classifier on Metrica Games 1 and 2. Metrica: Games 1
and 2 x seeds 0-2, classifier on the other game. Game 3 is locked, the challenge match is not read.

`check` trains with load_tracking and graded_mask made to raise and the audit guard on, then checks that
own_gap_share in degrade gives the option C tracks. `count` compares two submission files, with no scoring.
"""
import no_challenge_read as N      # audit guard first
import csv
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
WORKERS = 4
SHARE = 0.2
SPREAD = {}


def read_labels(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    col = rows[0].index("label") if "label" in rows[0] else len(rows[0]) - 1
    return rows[0], [r[0] for r in rows[1:]], [r[col] for r in rows[1:]]


def count(a, b):
    ha, fa, la = read_labels(a)
    hb, fb, lb = read_labels(b)
    assert ha == hb and fa == fb, "the two files differ in header or frames"
    moved = Counter((x, y) for x, y in zip(la, lb) if x != y)
    print("frames {}, changed {} ({:.2%})".format(len(la), sum(moved.values()), sum(moved.values()) / len(la)))
    print("label counts current {}".format(dict(sorted(Counter(la).items()))))
    print("label counts new     {}".format(dict(sorted(Counter(lb).items()))))
    for (x, y), n in sorted(moved.items()):
        print("  {} -> {}  {}".format(x, y, n))
    return moved


def check():
    import numpy as np
    from possession import deadball, external, validate as V

    def refuse(*a, **k):
        raise RuntimeError("the challenge match was read during training")
    real = external.load_tracking, external.graded_mask
    external.load_tracking = external.graded_mask = refuse
    V.lock_holdout()
    t0 = time.time()
    models = deadball.train_spread(V.TUNE, seed=0)
    print("train_spread on {} with the read refused: fine, {:.0f}s".format(V.TUNE, time.time() - t0))
    x = np.random.default_rng(0).normal(size=(1000, models[1].n_features_in_)).astype(np.float32)
    print("players-only p sum {:.6f}".format(models[1].predict_proba(x)[:, 1].sum()))
    for name in V.TUNE:
        clean, raw, _ = external._clean_game(name)
        ctl = external.require_control(raw, external.bench_events(name, clean["bx"]), external.CONTROL_SECONDS)
        assert np.array_equal(raw == "D", ctl == "D") and np.array_equal(raw == "X", ctl == "X"), name
    print("dead and X frames are the same with and without the control rule")
    external.load_tracking, external.graded_mask = real

    import gap_options as G                 # installs the stubs and its own degrade
    for name, seed in ((V.TUNE[0], 2), (V.TUNE[1], 1)):
        t, _, _ = G.load(name, seed, SHARE)
        clean, raw, _ = external._clean_game(name)
        mine = G._REAL_DEGRADE(clean, seed=seed, labels=raw, waiting_share=external.WAITING_SHARE,
                               after_whistle_share=external.AFTER_WHISTLE_SHARE,
                               fill_frames=external.BALL_FILL_FRAMES, own_gap_share=SHARE)
        same = all(np.array_equal(t[k], mine[k], equal_nan=True) for k in ("bx", "by", "px", "py", "ps"))
        print("{} seed {}: own_gap_share gives the option C tracks: {}".format(name, seed, same))
        assert same


def _spread(key):
    import gap_options as G
    from possession import deadball
    pool, seed = key
    G.set_share(0.0)        # the bench's extra gaps stay out of training
    return key, deadball.train_spread(list(pool), seed=seed)


def job(args):
    import gap_options as G
    from possession import deadball, external, model
    np = G.np
    name, seed = args
    t, labels, blocks = G.load(name, seed, SHARE)
    pool = G.METRICA if name in G.DFL else G.pool_of(name)
    assert name not in pool
    rest = G.rest_fn(name, t, seed)
    first = next(iter(labels.values()))
    res, clf, preds = {}, {}, {}
    for tag, m in (("option_c", G.model_for(SHARE, name, seed)), ("spread", SPREAD[pool, seed])):
        p = deadball.p_dead(m, t, external.PITCH)
        res[tag], preds[tag] = G.score_all(t, labels, blocks, rest, p, model.PARAMS)
        clf[tag] = G.clf_metrics(m, t, first, p, seed)
    g = first != "X"
    return dict(name=name, seed=seed, res=res, clf=clf,
                frames_moved=float((preds["option_c"][g] != preds["spread"][g]).mean()))


def bench(out_path, one=None):
    import gap_options as G                 # installs the stubs, locks the holdout
    np, V = G.np, G.V
    t0 = time.time()
    todo = [one] if one else G.runs()
    keys = sorted({(G.METRICA if n in G.DFL else G.pool_of(n), s) for n, s in todo}, key=str)
    G.train_all([(SHARE, p, s, False) for p, s in keys])
    G.set_share(0.0)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for key, m in pool.imap_unordered(_spread, keys):
            SPREAD[key] = m
    print("[{:5.0f}s] classifiers trained".format(time.time() - t0), flush=True)
    jobs = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for j in pool.imap_unordered(job, todo):
            jobs.append(j)
            json.dump(dict(jobs=jobs), open(out_path, "w"))
            print("[{:5.0f}s] {}/{} {} seed {}".format(time.time() - t0, len(jobs), len(todo), j["name"], j["seed"]), flush=True)
    jobs.sort(key=lambda j: (j["name"] not in G.DFL, j["name"], j["seed"]))
    if one:
        print(json.dumps(jobs[0], indent=1))
        return
    summary = {}
    for b, c, rows in (("DFL 1.5 s", "1.5", [j for j in jobs if j["name"] in G.DFL]),
                       ("DFL 2.5 s", "2.5", [j for j in jobs if j["name"] in G.DFL]),
                       ("Metrica", "metrica", [j for j in jobs if j["name"] not in G.DFL])):
        val = lambda tag, k="spell_f1_mean": [j["res"][tag][c][k] for j in rows]
        g = list(np.array(val("spread")) - np.array(val("option_c")))
        ok, why = V.verdict(g)
        summary[b] = dict(n=len(g), option_c=float(np.mean(val("option_c"))), spread=float(np.mean(val("spread"))),
                          mean_gain=float(np.mean(g)), up=int(sum(x > 0 for x in g)), down=int(sum(x < 0 for x in g)),
                          min_gain=float(min(g)), macro_c=float(np.mean(val("option_c", "macro_f1"))),
                          macro_s=float(np.mean(val("spread", "macro_f1"))),
                          spells_c=float(np.mean(val("option_c", "n_pred_spells"))),
                          spells_s=float(np.mean(val("spread", "n_pred_spells"))),
                          auc_c=float(np.mean([j["clf"]["option_c"]["auc"] for j in rows])),
                          auc_s=float(np.mean([j["clf"]["spread"]["auc"] for j in rows])),
                          moved=float(np.mean([j["frames_moved"] for j in rows])), passed=bool(ok), why=why)
    json.dump(dict(jobs=jobs, summary=summary, wall_s=time.time() - t0), open(out_path, "w"), indent=1)
    print("\n  {:9s} {:>4s} {:>8s} {:>8s} {:>8s} {:>3s} {:>4s} {:>8s} {:>13s} {:>15s} {:>13s} {:>6s}  {}".format(
        "bench", "runs", "option_c", "spread", "gain", "up", "down", "min", "macro c/s", "spells c/s", "auc c/s", "moved", "strict rule"))
    for b, s in summary.items():
        print("  {:9s} {:4d} {:8.4f} {:8.4f} {:+8.4f} {:3d} {:4d} {:+8.4f} {:6.4f}/{:6.4f} {:7.1f}/{:7.1f} {:6.4f}/{:6.4f} {:6.3f}  {}".format(
            b, s["n"], s["option_c"], s["spread"], s["mean_gain"], s["up"], s["down"], s["min_gain"], s["macro_c"], s["macro_s"],
            s["spells_c"], s["spells_s"], s["auc_c"], s["auc_s"], s["moved"], "pass" if s["passed"] else "fail: " + s["why"]))
    print("\nGain per run")
    for j in jobs:
        print("  {:14s} seed {}  ".format(j["name"], j["seed"]) + "  ".join(
            "{} {:.4f} {:+.4f}".format(c, j["res"]["option_c"][c]["spell_f1_mean"],
                                       j["res"]["spread"][c]["spell_f1_mean"] - j["res"]["option_c"][c]["spell_f1_mean"])
            for c in j["res"]["option_c"]))
    print("\nwall time {:.0f}s".format(time.time() - t0))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "check":
        check()
    elif mode == "bench":
        bench(sys.argv[2], (sys.argv[3], int(sys.argv[4])) if len(sys.argv) > 3 else None)
    else:
        count(sys.argv[2], sys.argv[3])
