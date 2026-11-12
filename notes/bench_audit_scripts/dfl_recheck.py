"""One-change re-check of model settings on the DFL open data, for two label versions.

    MTAH_DFL_DIR=/path OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 .venv/bin/python notes/bench_audit_scripts/dfl_recheck.py OUT.json
    ... dfl_recheck.py OUT.json MATCH SEED CONTROL [candidate label ...]    # one job, for a determinism check

Seven matches x seeds 0, 1, 2 x control 1.5 s and 2.5 s. DFL is scoring only: the dead ball classifier
is trained on Metrica Games 1 and 2 (training labels stay at external.CONTROL_SECONDS) and the Metrica
holdout is locked. Scoring is validate._score_one, with the DFL game loaded at the given control length.
Every candidate differs from model.PARAMS in one key. A version passes when all 21 gains are above 0
and their mean is at least +0.005 (validate.verdict). Test only; nothing here changes a setting.
"""
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import numpy as np
from possession_metrics import evaluate, label_codes
from possession import deadball, external, external_dfl, model, restarts, validate as V

PITCH = external.PITCH
SEEDS = (0, 1, 2)
CONTROLS = (1.5, 2.5)
KEYS = ("spell_f1_mean", "macro_f1", "n_pred_spells")
WORKERS = 5

OFF = [("dead_gap_features", ["ball", "team", "all"]), ("kickoff_seen_only", [True]),
       ("goal_kick_exit", [8]), ("play_on_speed", [1.4]), ("fast_unseen", [6]),
       ("restart_fill", [5]), ("restart_alive", [12, 25]), ("restart_dead_before", [25, 50]),
       ("min_live_in_dead", [50, 125]), ("contested_keeps", [True]),
       ("confirm", ["contested", "all"]), ("switch_control", [5, 10]), ("follow_weight", [0.25]),
       ("reception_path", [True])]
MODEL = [("switch_team", [40, 80]), ("switch_touch", [7, 14]), ("switch_dead", [5, 20]),
         ("touch_jerk", [1.0, 2.5]), ("touch_reach", [2.0, 3.0]), ("touch_lag", [0, 5]),
         ("touch_window", [1, 5]), ("touch_hold", [125, 500]), ("touch_weight", [0.125, 0.5]),
         ("touch_direction", [False]), ("touch_angle", [5, 20]), ("touch_dspeed", [3, 8]),
         ("scuffle_radius", [0, 3]), ("min_dead", [50, 100]), ("out_margin", [0, 1]),
         ("slow_speed", [1.2, 2.0]), ("missing_long", [25, 100]), ("blip_max_speed", [0, 3, 6]),
         ("blip_max_len", [125, 500]), ("use_restarts", [False]), ("restart_frames", [6, 25]),
         ("restart_weight", [1, 4]), ("throw_ins", [False]), ("restart_moving", [4, 8]),
         ("keep_restart_spells", [False])]
REPORT = [("min_spell", [0, 25, 38, 63, 75]), ("spell_keep_touches", [2])]


def candidates():
    """List of (group, label, params); keys missing from model.PARAMS are skipped and reported."""
    out, missing = [], []
    for group, items in (("off", OFF), ("model", MODEL), ("report", REPORT)):
        for key, values in items:
            if key not in model.PARAMS:
                missing.append(key)
                continue
            for v in values:
                p = dict(model.PARAMS, **{key: v})
                diff = [k for k in p if p[k] != model.PARAMS[k] or type(p[k]) is not type(model.PARAMS[k])]
                assert diff == [key], (key, v, diff)
                out.append((group, "{}={}".format(key, v), p))
    return out, missing


def gap_of(p):
    return p.get("dead_gap_features") or False


def train_all(cands):
    """Dead ball classifiers on Metrica Games 1 and 2, one per (seed, gap), cached in validate._DEAD."""
    gaps = {gap_of(model.PARAMS)} | {gap_of(p) for _, _, p in cands}
    for seed in SEEDS:
        for gap in sorted(gaps, key=str):
            V._dead_models(V.TUNE, seed, gap)


def run_job(args):
    name, seed, control, cands, wanted = args
    t0, labels, blocks = external_dfl.load_game(name, seed=seed, control=control)
    codes_true = label_codes(labels)
    tags = restarts.simulated_tags(external_dfl.events_for(name), seed=seed)
    dead, rest = {}, {}

    def score(params):
        t = t0
        if params.get("learned_dead"):
            gap = gap_of(params)
            if gap not in dead:
                dead[gap] = deadball.p_dead(V._dead_models(V.TUNE, seed, gap), t0, PITCH)
            t = dict(t, p_dead=dead[gap])
        if params.get("use_restarts"):
            k = (bool(params.get("throw_ins")), params.get("restart_moving", 6.0))
            if k not in rest:
                rest[k] = restarts.restart_events(t0, tags, pitch=PITCH if k[0] else None, moving=k[1])
            t = dict(t, restarts=rest[k])
        pred = model.predict(t, PITCH, blocks, len(labels), params)
        r = evaluate(codes_true, label_codes(pred), bootstrap_n=0)
        return {k: float(r[k]) for k in KEYS}, int(r["n_gt_spells"])

    res = {}
    res["ref"], n_gt = score(model.PARAMS)
    for group, label, p in cands:
        if wanted is None or label in wanted:
            res[label] = score(p)[0]
    digest = hashlib.md5(np.ascontiguousarray(labels).tobytes()).hexdigest()
    return dict(name=name, seed=seed, control=control, res=res, n_gt=n_gt, labels_md5=digest)


def summarize(jobs, cands):
    """Per control, per candidate: the gain statistics over all runs, and the verdict."""
    out = {}
    for control in CONTROLS:
        rows = [j for j in jobs if j["control"] == control]
        if len(rows) != len(external_dfl.GAMES) * len(SEEDS):
            continue
        out[str(control)] = {}
        for group, label, p in cands:
            g = [j["res"][label]["spell_f1_mean"] - j["res"]["ref"]["spell_f1_mean"] for j in rows]
            m = [j["res"][label]["macro_f1"] - j["res"]["ref"]["macro_f1"] for j in rows]
            ok, why = V.verdict(g)
            out[str(control)][label] = dict(
                group=group, mean_gain=float(np.mean(g)), up=sum(x > 0 for x in g), down=sum(x < 0 for x in g),
                tied=sum(x == 0 for x in g), min_gain=float(min(g)), macro_gain=float(np.mean(m)),
                mean_pred_spells=float(np.mean([j["res"][label]["n_pred_spells"] for j in rows])),
                pass_=bool(ok), why=why)
    return out


def print_report(jobs, cands, summary):
    print("\nReference (model.PARAMS), mean over seeds")
    for control in CONTROLS:
        print("control {}".format(control))
        print("  {:8s} {:>9s} {:>9s} {:>10s} {:>9s}".format("match", "spell_f1", "macro_f1", "pred_spl", "true_spl"))
        for name in list(external_dfl.GAMES) + ["ALL"]:
            rows = [j for j in jobs if j["control"] == control and name in (j["name"], "ALL")]
            if rows:
                print("  {:8s} {:9.4f} {:9.4f} {:10.1f} {:9.1f}".format(
                    name, *[np.mean([j["res"]["ref"][k] for j in rows]) for k in KEYS[:3]],
                    np.mean([j["n_gt"] for j in rows])))
        same = all(len({j["labels_md5"] for j in jobs if j["control"] == control and j["name"] == n}) == 1
                   for n in external_dfl.GAMES)
        print("  labels identical across seeds for every match: {}".format(same))
    for control, table in summary.items():
        print("\nControl {}: gain over the reference, 21 runs".format(control))
        print("  {:7s} {:30s} {:>8s} {:>3s} {:>4s} {:>4s} {:>8s} {:>8s} {:>8s}  {}".format(
            "group", "candidate", "mean", "up", "down", "tied", "min", "macro", "pred_spl", "verdict"))
        def order(kv):
            worst = min(summary[c][kv[0]]["mean_gain"] for c in summary)
            return ["off", "model", "report"].index(kv[1]["group"]), worst
        for label, s in sorted(table.items(), key=order):
            print("  {:7s} {:30s} {:+8.4f} {:3d} {:4d} {:4d} {:+8.4f} {:+8.4f} {:8.1f}  {}".format(
                s["group"], label, s["mean_gain"], s["up"], s["down"], s["tied"], s["min_gain"],
                s["macro_gain"], s["mean_pred_spells"], "PASS" if s["pass_"] else "FAIL " + s["why"]))
    if len(summary) == 2:
        both = [l for l in summary["1.5"] if summary["1.5"][l]["pass_"] and summary["2.5"][l]["pass_"]]
        print("\nOverall PASS on both versions: {}".format(both or "none"))


def main():
    out_path = sys.argv[1]
    cands, missing = candidates()
    t_start = time.time()
    if len(sys.argv) > 2:      # single job
        name, seed, control = sys.argv[2], int(sys.argv[3]), float(sys.argv[4])
        V.lock_holdout()
        wanted = set(sys.argv[5:])
        train_all(cands)
        job = run_job((name, seed, control, cands, wanted))
        json.dump(job, open(out_path, "w"), indent=1)
        print(json.dumps(job, indent=1))
        return
    V.lock_holdout()
    train_all(cands)
    print("{} candidates; keys not in model.PARAMS (skipped): {}".format(len(cands), missing or "none"), flush=True)
    todo = [(n, s, c, cands, None) for c in CONTROLS for n in external_dfl.GAMES for s in SEEDS]
    jobs = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for job in pool.imap_unordered(run_job, todo):
            jobs.append(job)
            json.dump(dict(jobs=jobs, missing=missing), open(out_path, "w"))
            print("[{:5.0f}s] {}/{} {} seed {} control {} ref spellF1 {:.4f}".format(
                time.time() - t_start, len(jobs), len(todo), job["name"], job["seed"], job["control"],
                job["res"]["ref"]["spell_f1_mean"]), flush=True)
    summary = summarize(jobs, cands)
    json.dump(dict(jobs=jobs, missing=missing, summary=summary,
                   candidates=[(g, l) for g, l, _ in cands], wall_s=time.time() - t_start,
                   overall_pass=[l for l in summary.get("1.5", {}) if all(summary[c][l]["pass_"] for c in summary)]
                   if len(summary) == 2 else []), open(out_path, "w"), indent=1)
    print_report(jobs, cands, summary)
    print("\nwall time {:.0f}s".format(time.time() - t_start))


if __name__ == "__main__":
    main()
