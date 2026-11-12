"""Five extra features for the learned team evidence, one at a time, test only. Nothing is wired in.

Reference: Ramith's prototype (notes/evidence/team_evidence_prototype.py) in "added" mode at weight 0.25,
features, classifier settings (HistGradientBoosting, depth 4, 200 iterations, learning rate 0.05,
random_state = seed) and every 5th live frame with the ball seen, unchanged. As in
team_evidence_more_data.py's Metrica job (its "one_game" path): trained on the other Metrica game only,
model.PARAMS with restarts on, dead ball classifier trained on the other game. Games 1-2 x seeds 0-2,
clean leave-one-out; Game 3 is locked and the read guard is on, so our match is never opened and the
bench copies none of its missing-ball gaps.

Each candidate is the prototype's features plus one group below, same everything else. Every definition
was fixed before any scoring; nothing below is tuned. Rolling and trajectory features look back only.
Distances and densities do not depend on the attacking direction; the one directional feature (4a) is
turned so that + is towards the goal home attacks, read per half from the tracking (the team whose
players stand further left on average attacks right).

  1 density    Gaussian-weighted count of each team's players around the ball, sigma 5 m, and the
               home minus away difference (3)
  2 alignment  for each team's nearest player: cosine between their velocity and the ball's (0 if either
               moves under 0.5 m/s), speed relative to the ball, closing speed on the ball; velocities over
               the last 0.2 s (6)
  3 advantage  (da - dh) / (da + dh), log((da + 0.5) / (dh + 0.5)), second-nearest away minus home (3)
  4 trajectory over the last 0.4 s: a) ball displacement towards home's attacking goal, b) sideways
               displacement size, c) straightness (displacement over path length), d) away minus home
               nearest distance at the point the ball reaches 0.4 s ahead at that velocity (4)
  5 pressure   trailing means over the last 1 s and 3 s of the density difference (1) and of
               sign(da - dh), who was nearer (4)

Pass rule (fixed before scoring): a group passes if against the prototype it gains spellF1 on all 6 runs
with mean gain at least +0.005 (validate.verdict). The combination is every group that passes; with fewer
than two passing groups there is no combination. All five together are scored as a diagnostic only, never
a candidate. Also reported per run: macroF1, transition F1 (25 frames), predicted spells, and the
classifier's log loss and AUC on the scored game's live frames with the ball seen.

    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 .venv/bin/python notes/evidence/team_evidence_features.py OUT.json
    ... team_evidence_features.py OUT.json Sample_Game_1 0     # one job, for a determinism check
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "notes", "bench_audit_scripts"))
import no_challenge_read as N  # noqa: E402  audit guard first
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

os.chdir(ROOT)                 # the prototype puts the working directory on sys.path
sys.path.insert(0, os.path.join(ROOT, "notes", "evidence"))
import team_evidence_prototype as T  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402
from possession import deadball, external, model, restarts, validate as V  # noqa: E402

PITCH = external.PITCH
P = model.PARAMS
WEIGHT = 0.25
FPS = 25
GROUPS = ("density", "alignment", "advantage", "trajectory", "pressure")
WORKERS = 3
ROWS = {}                      # (match, seed) -> training rows of every feature, filled before the fork


def back(x, k):
    """x shifted k frames back, the first frames repeating frame 0."""
    return np.concatenate([np.repeat(x[:1], k, axis=0), x[:-k]])


def trailing(x, ok, w):
    """Mean of x over the last w frames (this one included) where ok."""
    c = np.r_[0, np.cumsum(np.where(ok, x, 0.0))]
    m = np.r_[0, np.cumsum(ok)]
    i = np.arange(1, len(x) + 1)
    j = np.maximum(i - w, 0)
    num, den = c[i] - c[j], m[i] - m[j]
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)


def attack_sign(t, blocks, n):
    """+1 where home attacks towards +x, -1 where towards -x, 0 outside the halves."""
    s = np.zeros(n)
    for _, a, b in blocks:
        px = t["px"][a:b + 1]
        hx = np.nanmean(px[:, t["team"] == 0])
        ax = np.nanmean(px[:, t["team"] == 1])
        s[a:b + 1] = 1.0 if hx < ax else -1.0
    return s


def extra(t, blocks, p=P):
    """{group: (n, k) array} of the five candidate groups."""
    bx, by = model.fill_ball(t["bx"], t["by"], p["max_gap"])
    n = len(bx)
    ok = np.isfinite(bx)
    d, tf, dh, da = model.team_distances(t, bx, by)
    out = {}

    dens = [np.where(tf == k, np.exp(-np.where(np.isfinite(d), d, 1e3) ** 2 / (2 * 5.0 ** 2)), 0).sum(1)
            for k in (0, 1)]
    out["density"] = np.column_stack([dens[0], dens[1], dens[0] - dens[1]])

    px, py = t["px"], t["py"]
    vpx, vpy = (px - back(px, 5)) * FPS / 5, (py - back(py, 5)) * FPS / 5
    vbx, vby = (bx - back(bx, 5)) * FPS / 5, (by - back(by, 5)) * FPS / 5
    vb = np.hypot(vbx, vby)
    rows = np.arange(n)
    dprev = back(d, 5)
    cols = []
    for k in (0, 1):
        j = np.argmin(np.where(tf == k, d, np.inf), axis=1)
        ux, uy = vpx[rows, j], vpy[rows, j]
        up = np.hypot(ux, uy)
        cos = np.where((up > 0.5) & (vb > 0.5), (ux * vbx + uy * vby) / np.maximum(up * vb, 1e-6), 0.0)
        rel = np.hypot(ux - vbx, uy - vby)
        close = -(d[rows, j] - dprev[rows, j]) * FPS / 5
        cols += [cos, rel, close]
    out["alignment"] = np.column_stack(cols)

    h, a = np.minimum(dh, 60), np.minimum(da, 60)
    s2 = [np.sort(np.where(tf == k, d, np.inf), axis=1)[:, 1] for k in (0, 1)]
    out["advantage"] = np.column_stack([(a - h) / np.maximum(a + h, 1e-6), np.log((a + 0.5) / (h + 0.5)),
                                        np.minimum(s2[1], 60) - np.minimum(s2[0], 60)])

    sign = attack_sign(t, blocks, n)
    dx, dy = bx - back(bx, 10), by - back(by, 10)
    step = np.hypot(np.diff(bx, prepend=bx[0]), np.diff(by, prepend=by[0]))
    path = trailing(step, np.isfinite(step), 10) * 10
    straight = np.where(path > 0.1, np.hypot(dx, dy) / np.maximum(path, 1e-6), 0.0)
    _, _, eh, ea = model.team_distances(t, bx + dx, by + dy)
    out["trajectory"] = np.column_stack([sign * dx, np.abs(dy), np.minimum(straight, 1.0),
                                         np.minimum(ea, 60) - np.minimum(eh, 60)])

    diff = dens[0] - dens[1]
    near = np.sign(a - h)
    out["pressure"] = np.column_stack([trailing(diff, ok, FPS), trailing(diff, ok, 3 * FPS),
                                       trailing(near, ok, FPS), trailing(near, ok, 3 * FPS)])
    return {g: np.nan_to_num(v, nan=0.0, posinf=60.0, neginf=-60.0).astype(np.float32) for g, v in out.items()}


def all_features(t, blocks):
    F, ok = T.feats(t)
    return F, extra(t, blocks), ok


def stack(F, E, groups):
    return np.column_stack([F] + [E[g] for g in groups]) if groups else F


def rows(args):
    name, seed = args
    t, y, blocks = external.load_game(name, seed=seed)
    F, E, ok = all_features(t, blocks)
    idx = np.flatnonzero(np.isin(y, ["H", "A"]) & ok)[::5]
    return args, F[idx], {g: v[idx] for g, v in E.items()}, (np.asarray(y)[idx] == "H").astype(int)


def fit(other, seed, groups):
    F, E, y = ROWS[other, seed]
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
    return clf.fit(stack(F, E, groups), y)


def predict_added(t, blocks, n):
    emis, orig = T.make_emissions("added", WEIGHT)
    model.emissions = emis
    try:
        return model.predict(t, PITCH, blocks, n, P)
    finally:
        model.emissions = orig


def scores(labels, pred):
    r = evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)
    return dict(spell_f1=float(r["spell_f1_mean"]), macro_f1=float(r["macro_f1"]),
                trans_f1=float(r["transitions"][25]["f1"]), n_pred_spells=int(r["n_pred_spells"]),
                n_gt_spells=int(r["n_gt_spells"]))


def candidates(passing):
    c = {"proto": ()}
    c.update({g: (g,) for g in GROUPS})
    if passing is not None and len(passing) >= 2:
        c["combo"] = tuple(passing)
    if passing is None:
        c["all5"] = GROUPS
    return c


def job(args):
    name, seed, sets = args
    t0 = time.time()
    other = [g for g in V.TUNE if g != name][0]
    tr, y, blocks = external.load_game(name, seed=seed)
    n = len(y)
    t = dict(tr, p_dead=deadball.p_dead(V._dead_models((other,), seed, False), tr, PITCH))
    tags = restarts.simulated_tags(external.events_for(name), seed=seed)
    t = dict(t, restarts=restarts.restart_events(t, tags, pitch=PITCH if P.get("throw_ins") else None,
                                                  moving=P.get("restart_moving", 6.0)))
    F, E, ok = all_features(t, blocks)
    live = np.isin(y, ["H", "A"]) & ok
    yh = (np.asarray(y)[live] == "H").astype(int)
    res = {"ref": scores(y, model.predict(t, PITCH, blocks, n, P))}
    for key, groups in sets.items():
        clf = fit(other, seed, groups)
        ph = clf.predict_proba(stack(F, E, groups))[:, 1]
        r = scores(y, predict_added(dict(t, p_team=ph, p_ok=ok), blocks, n))
        r.update(log_loss=float(log_loss(yh, np.clip(ph[live], 1e-6, 1 - 1e-6))),
                 auc=float(roc_auc_score(yh, ph[live])))
        res[key] = r
    return dict(name=name, seed=seed, train=other, res=res, job_s=time.time() - t0)


def gains(jobs, a, b, k="spell_f1"):
    return [j["res"][a][k] - j["res"][b][k] for j in jobs]


def report(jobs, keys):
    jobs = sorted(jobs, key=lambda j: (j["name"], j["seed"]))
    print("\nper run spellF1 (reference = model.PARAMS without team evidence)")
    print("  {:14s} {:4s} ".format("match", "seed") + " ".join("{:>10s}".format(k) for k in ["ref"] + keys))
    for j in jobs:
        print("  {:14s} {:4d} ".format(j["name"], j["seed"]) +
              " ".join("{:10.4f}".format(j["res"][k]["spell_f1"]) for k in ["ref"] + keys))
    print("\n  {:10s} {:>9s} {:>9s} {:>5s} {:>9s} {:>9s} {:>9s} {:>9s} {:>7s} {:>7s} {:>6s}  {}".format(
        "set", "vs ref", "vs proto", "up", "min", "macro", "trans", "spells", "logloss", "AUC", "", "verdict vs proto"))
    summary = {}
    for k in keys:
        g = gains(jobs, k, "proto")
        ok, why = V.verdict(g) if k != "proto" else (False, "reference")
        s = dict(vs_ref=float(np.mean(gains(jobs, k, "ref"))), vs_proto=float(np.mean(g)),
                 up=int(sum(x > 0 for x in g)), min=float(min(g)),
                 macro=float(np.mean(gains(jobs, k, "proto", "macro_f1"))),
                 trans=float(np.mean(gains(jobs, k, "proto", "trans_f1"))),
                 spells=float(np.mean([j["res"][k]["n_pred_spells"] for j in jobs])),
                 log_loss=float(np.mean([j["res"][k]["log_loss"] for j in jobs])),
                 auc=float(np.mean([j["res"][k]["auc"] for j in jobs])), passed=bool(ok), why=why,
                 per_run=g)
        summary[k] = s
        print("  {:10s} {:+9.4f} {:+9.4f} {:3d}/6 {:+9.4f} {:+9.4f} {:+9.4f} {:9.1f} {:7.4f} {:7.4f} {:>6s}  {}".format(
            k, s["vs_ref"], s["vs_proto"], s["up"], s["min"], s["macro"], s["trans"], s["spells"], s["log_loss"],
            s["auc"], "", "-" if k == "proto" else ("PASS" if ok else "FAIL " + why)))
    pr = gains(jobs, "proto", "ref")
    print("\nprototype vs reference: mean {:+.4f}, up {}/6, per run {}".format(
        np.mean(pr), sum(x > 0 for x in pr), " ".join("{:+.4f}".format(x) for x in pr)))
    print("reference mean spellF1 {:.4f}, true spells per run {:.1f}, reference spells {:.1f}".format(
        np.mean([j["res"]["ref"]["spell_f1"] for j in jobs]), np.mean([j["res"]["ref"]["n_gt_spells"] for j in jobs]),
        np.mean([j["res"]["ref"]["n_pred_spells"] for j in jobs])))
    return summary


def main():
    out_path = sys.argv[1]
    N.install()
    V.lock_holdout()
    t_start = time.time()
    single = len(sys.argv) > 2
    seeds = (int(sys.argv[3]),) if single else V.SEEDS
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for key, F, E, y in pool.imap_unordered(rows, [(g, s) for g in V.TUNE for s in seeds]):
            ROWS[key] = F, E, y
    for s in seeds:            # dead ball classifiers, inherited by the workers
        for g in V.TUNE:
            V._dead_models(tuple(x for x in V.TUNE if x != g), s, False)
    print("[{:4.0f}s] training rows ready".format(time.time() - t_start), flush=True)
    if single:
        j = job((sys.argv[2], int(sys.argv[3]), candidates(None)))
        json.dump(j, open(out_path, "w"), indent=1)
        print(json.dumps({k: v["spell_f1"] for k, v in j["res"].items()}))
        return
    todo = [(g, s) for g in V.TUNE for s in V.SEEDS]
    stage1 = dict(candidates(None))
    jobs = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for j in pool.imap_unordered(job, [(g, s, stage1) for g, s in todo]):
            jobs.append(j)
            print("[{:4.0f}s] {} seed {} ({:.0f}s)".format(time.time() - t_start, j["name"], j["seed"], j["job_s"]),
                  flush=True)
    print("\nStage 1: each group alone (all5 is a diagnostic, not a candidate)")
    summary = report(jobs, ["proto"] + list(GROUPS) + ["all5"])
    passing = [g for g in GROUPS if summary[g]["passed"]]
    print("\npassing groups: {}".format(", ".join(passing) or "none"))
    combo = None
    if len(passing) >= 2:
        sets = {"proto": (), "combo": tuple(passing)}
        cj = []
        with mp.get_context("fork").Pool(WORKERS) as pool:
            for j in pool.imap_unordered(job, [(g, s, sets) for g, s in todo]):
                cj.append(j)
        print("\nStage 2: combination of {}".format(", ".join(passing)))
        combo = dict(jobs=cj, summary=report(cj, ["proto", "combo"]))
    else:
        print("fewer than two groups pass, so there is no combination")
    json.dump(dict(jobs=jobs, summary=summary, passing=passing, combo=combo, wall_s=time.time() - t_start),
              open(out_path, "w"), indent=1)
    print("\nwall time {:.0f}s".format(time.time() - t_start))


if __name__ == "__main__":
    main()
