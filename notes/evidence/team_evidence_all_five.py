"""The five extra features on all three benches, two candidates, test only. Nothing is wired in.

Runs the pre-registration in notes/proposals/team_evidence_all_five.md. Not to be run before Ramith and
Aditya have agreed to that text.

    MTAH_DFL_DIR=/path OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONHASHSEED=0 .venv/bin/python notes/evidence/team_evidence_all_five.py OUT.json
    ... team_evidence_all_five.py OUT.json MATCH SEED     # one job, for the determinism check
    ... team_evidence_all_five.py check                   # no candidate is scored, see below

Bench, the current one (notes/bench_audit_scripts/gap_options.py, spread_training.py; the same as the
verified notes/evidence/control_sweep.py): option C tracks (the match's own gaps at share 0.2), dead ball
classifier deadball.train_spread, Metrica control 1.5 s (external.CONTROL_SECONDS, asserted). DFL: 7 matches x
seeds 0-2 at control 1.5 s and 2.5 s, dead ball classifier on Metrica Games 1 and 2. Metrica: Games 1 and 2 x
seeds 0-2, dead ball classifier on the other game. Game 3 is locked, the challenge match is not read.
Before the verdict is printed the mean reference spellF1 is asserted: Metrica 0.6153, DFL 1.5 s 0.5140, DFL
2.5 s 0.5624 (abs tol 1e-4).

Training rows of the team classifier: the tracks of share 0.0 (no own gaps), as gap_options/spread_training
train their dead ball classifiers (set_share(0.0): the bench's extra gaps stay out of training). The
scored match gets its own gaps at 0.2; its classifier never saw them. DFL labels at the 1.5 s control.

Differences from notes/bench_audit_scripts/team_evidence_more_data.py:
  1 features: the prototype's (team_evidence_prototype.feats) plus all five groups of
    team_evidence_features.extra, unchanged (feats)
  2 weight 0.25 for the "added" candidate, fixed (no TEAM_WEIGHT)
  3 a second candidate, the spell vote (spell_vote below, notes/proposals/team_evidence_spell_vote.md)
  4 the bench: the current one above, instead of plain load_game tracks and deadball.train
  5 dropped: the no-restarts prototype path (proto_ref, proto_new, proto_path in stats and report)
  6 restructured: summarize, stats and print_report run over BENCHES and SIDE, per candidate, with the
    decision rule; scores() reports transition F1 instead of KEYS; the import prologue is different (HERE,
    team_evidence_features, sklearn.metrics, gap_options, spread_training, noqa marks)
  7 new helpers: with_restarts, load, live_spells, classifier_fit, candidates, and the check and
    reproduction code (old_metrica_job, reproduce, reference_check, check)
Training pools of the team classifier, classifier settings, step 5, restarts, the reference and the guards
are as there. Added for reporting only: transition F1, the classifier's log loss and AUC (per control
length on DFL, against the labels of the row), the share of spells the vote flips, and for Metrica the scores
against the labels without the control rule (CONTROL_SECONDS = 0, which equal external._clean_game's
labels, PR 29) and the one-game path (other Metrica game only; its classifier's own log loss, AUC and flips).

Reproduction check (the old bench, kept only here: plain load_game tracks, deadball.train on the other
game, one-game classifier): the one-game "added" scores on Metrica must equal the "all5" column of
notes/evidence/team_evidence_features.json run by run, and the reference 0.6685 must reproduce. It runs
first; if it fails the script raises before anything else is reported.

check: installs the guards, trains the dead ball classifiers, scores only the reference (model.PARAMS with
restarts, no team evidence) for all 27 runs, asserts the three references and CONTROL_SECONDS == 1.5, and
confirms team_evidence_features.json has the expected runs. No team classifier is fitted, no candidate
is predicted or scored.

Pass bar: a candidate passes if on DFL 1.5 s, DFL 2.5 s and Metrica every run gains over the reference and
the mean gain is at least +0.005 (validate.verdict). If both pass, the one with the higher lowest gain over
all 48 runs is kept.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "notes", "bench_audit_scripts"))
import no_challenge_read as N  # noqa: E402  audit guard first
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import time  # noqa: E402

os.chdir(ROOT)                 # the prototype puts the working directory on sys.path
sys.path.insert(0, HERE)
import team_evidence_prototype as T  # noqa: E402
import team_evidence_features as TF  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.metrics import log_loss, roc_auc_score  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402
from possession import deadball, external, external_dfl, model, restarts, validate as V  # noqa: E402
import gap_options as G  # noqa: E402  (stubs installed, Game 3 locked)
import spread_training as S  # noqa: E402

PITCH = external.PITCH
P = model.PARAMS
SEEDS = (0, 1, 2)
CONTROLS = (1.5, 2.5)
WEIGHT = 0.25
CANDS = ("added", "vote")
WORKERS = 4
DFL = external_dfl.GAMES
METRICA = V.TUNE
ROWS = {}                      # (match, seed) -> training rows, filled before the workers fork
REFERENCE = {"metrica": 0.6153, "1.5": 0.5140, "2.5": 0.5624}     # mean reference spellF1, current bench
REPRODUCTION = 0.6685          # mean reference spellF1 of the old bench, team_evidence_features.json
FEATURES_JSON = os.path.join(HERE, "team_evidence_features.json")


def feats(t, blocks):
    """The prototype's features plus all five extra groups, in a fixed order."""
    F, ok = T.feats(t)
    E = TF.extra(t, blocks)
    return TF.stack(F, E, TF.GROUPS), ok


def load(name, seed, share):
    """Tracks, {label version: labels}, blocks of the current bench (gap_options.load): own gaps at `share`;
    DFL labels at "1.5" and "2.5" s control, Metrica labels at CONTROL_SECONDS under "metrica"."""
    assert external.CONTROL_SECONDS == 1.5, external.CONTROL_SECONDS
    return G.load(name, seed, share)


def rows(args):
    name, seed = args
    t, labs, blocks = load(name, seed, 0.0)      # training rows: no own gaps, as the dead ball classifiers
    y = labs["1.5"] if name in DFL else labs["metrica"]
    F, ok = feats(t, blocks)
    idx = np.flatnonzero(np.isin(y, ["H", "A"]) & ok)[::5]
    return args, F[idx], (np.asarray(y)[idx] == "H").astype(int)


def pool_for(name):
    if name in DFL:
        return list(METRICA) + [g for g in DFL if g != name]
    return [g for g in METRICA if g != name] + list(DFL)


def fit(pool, seed, scored):
    assert scored not in pool, (scored, pool)
    X = np.concatenate([ROWS[g, seed][0] for g in pool])
    y = np.concatenate([ROWS[g, seed][1] for g in pool])
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
    return clf.fit(X, y), len(y)


def predict_added(t, blocks, n):
    emis, orig = T.make_emissions("added", WEIGHT)
    model.emissions = emis
    try:
        return model.predict(t, PITCH, blocks, n, P)
    finally:
        model.emissions = orig


def spell_vote(pred, p_home, p_ok):
    """The model's prediction with each live spell given to the team its mean P(home) favours.

    A spell is a maximal run of H or of A. Its mean is over its frames with the ball seen; a spell with none
    is left alone, and so is a spell that starts straight after a stoppage (the frame before is D or X, or
    it starts the match). Flips are decided on the model's spells, all at once. Returns the new labels and
    (spells flipped, spells considered)."""
    new = pred.copy()
    ch = np.flatnonzero(np.r_[True, pred[1:] != pred[:-1]])
    flipped = considered = 0
    for a, b in zip(ch, np.r_[ch[1:], len(pred)]):
        lab = pred[a]
        if lab not in "HA" or a == 0 or pred[a - 1] not in "HA":
            continue
        seen = p_ok[a:b]
        if not seen.any():
            continue
        considered += 1
        m = float(np.mean(p_home[a:b][seen]))
        if (lab == "H" and m < 0.5) or (lab == "A" and m > 0.5):
            new[a:b] = "A" if lab == "H" else "H"
            flipped += 1
    return new, (flipped, considered)


def live_spells(lab):
    g = lab[lab != "X"]
    return int(np.sum(np.r_[True, g[1:] != g[:-1]] & np.isin(g, ["H", "A"])))


def scores(labels, pred):
    r = evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)
    return dict(spell_f1_mean=float(r["spell_f1_mean"]), macro_f1=float(r["macro_f1"]),
                trans_f1=float(r["transitions"][25]["f1"]), n_pred_spells=float(r["n_pred_spells"]),
                n_gt_spells=int(r["n_gt_spells"]))


def candidates(t, F, ok, clf, blocks, n, base):
    """{"added": labels, "vote": labels}, the vote's flip counts, and P(home)."""
    ph = clf.predict_proba(F)[:, 1]
    added = predict_added(dict(t, p_team=ph, p_ok=ok), blocks, n)
    vote, flips = spell_vote(base, ph, ok)
    assert live_spells(vote) <= live_spells(base), "the spell vote added spells"
    return dict(added=added, vote=vote), flips, ph


def classifier_fit(labels, ph, ok):
    live = np.isin(labels, ["H", "A"]) & ok
    yh = (np.asarray(labels)[live] == "H").astype(int)
    return dict(log_loss=float(log_loss(yh, np.clip(ph[live], 1e-6, 1 - 1e-6))), auc=float(roc_auc_score(yh, ph[live])))


def with_restarts(name, t, seed):
    tags = restarts.simulated_tags(G.events(name), seed=seed)
    return dict(t, restarts=restarts.restart_events(t, tags, pitch=PITCH if P.get("throw_ins") else None,
                                                    moving=P.get("restart_moving", 6.0)))


def dfl_job(name, seed):
    t0, labs, blocks = load(name, seed, S.SHARE)
    n = len(labs["1.5"])
    t = dict(t0, p_dead=deadball.p_dead(S.SPREAD[METRICA, seed], t0, PITCH))
    t = with_restarts(name, t, seed)
    F, ok = feats(t, blocks)
    pool = pool_for(name)
    clf, n_train = fit(pool, seed, name)
    base = model.predict(t, PITCH, blocks, n, P)
    cand, flips, ph = candidates(t, F, ok, clf, blocks, n, base)
    res = {c: dict(ref=scores(labs[c], base), **{k: scores(labs[c], v) for k, v in cand.items()}) for c in labs}
    return dict(kind="dfl", name=name, seed=seed, pool=pool, n_train=n_train, res=res,
                flips={c: flips for c in labs}, clf={c: classifier_fit(labs[c], ph, ok) for c in labs})


def metrica_job(name, seed):
    t0, labs, blocks = load(name, seed, S.SHARE)
    y = labs["metrica"]
    n = len(y)
    raw = external._clean_game(name)[1]             # the labels without the control rule
    other = G.pool_of(name)
    t = dict(t0, p_dead=deadball.p_dead(S.SPREAD[other, seed], t0, PITCH))
    t = with_restarts(name, t, seed)
    F, ok = feats(t, blocks)
    pool = pool_for(name)
    clf, n_train = fit(pool, seed, name)
    small, _ = fit(list(other), seed, name)
    base = model.predict(t, PITCH, blocks, n, P)
    cand, flips, ph = candidates(t, F, ok, clf, blocks, n, base)
    one, flips_one, ph_one = candidates(t, F, ok, small, blocks, n, base)
    res = {"metrica": dict(ref=scores(y, base), **{k: scores(y, v) for k, v in cand.items()}),
           "metrica_one_game": dict(ref=scores(y, base), **{k: scores(y, v) for k, v in one.items()}),
           "metrica_control_off": dict(ref=scores(raw, base), **{k: scores(raw, v) for k, v in cand.items()})}
    return dict(kind="metrica", name=name, seed=seed, pool=pool, n_train=n_train, res=res,
                flips={"metrica": flips, "metrica_one_game": flips_one, "metrica_control_off": flips},
                clf={"metrica": classifier_fit(y, ph, ok), "metrica_one_game": classifier_fit(y, ph_one, ok),
                     "metrica_control_off": classifier_fit(raw, ph, ok)})


def reference_job(name, seed):
    """The reference alone (model.PARAMS with restarts, no team evidence) on the current bench: for check."""
    t0, labs, blocks = load(name, seed, S.SHARE)
    pool = METRICA if name in DFL else G.pool_of(name)
    t = dict(t0, p_dead=deadball.p_dead(S.SPREAD[pool, seed], t0, PITCH))
    t = with_restarts(name, t, seed)
    base = model.predict(t, PITCH, blocks, len(next(iter(labs.values()))), P)
    return dict(name=name, seed=seed, ref={c: scores(lab, base)["spell_f1_mean"] for c, lab in labs.items()})


def old_metrica_job(name, seed):
    """The OLD bench, for the reproduction check only: plain load_game tracks (no own gaps), dead ball
    classifier deadball.train on the other game, classifier on the other game alone, "added" at 0.25."""
    G.set_share(0.0)
    t0, y, blocks = external.load_game(name, seed=seed)
    n = len(y)
    other = [g for g in METRICA if g != name]
    t = dict(t0, p_dead=deadball.p_dead(V._dead_models(tuple(other), seed, False), t0, PITCH))
    t = with_restarts(name, t, seed)
    F, ok = feats(t, blocks)
    small, _ = fit(other, seed, name)
    base = model.predict(t, PITCH, blocks, n, P)
    added = predict_added(dict(t, p_team=small.predict_proba(F)[:, 1], p_ok=ok), blocks, n)
    return dict(name=name, seed=seed, ref=scores(y, base)["spell_f1_mean"], added=scores(y, added)["spell_f1_mean"])


def run(args):
    t0 = time.time()
    job = dfl_job(*args) if args[0] in DFL else metrica_job(*args)
    job["job_s"] = time.time() - t0
    return job


BENCHES = (("DFL 1.5", "dfl", "1.5"), ("DFL 2.5", "dfl", "2.5"), ("Metrica", "metrica", "metrica"))
SIDE = (("Metrica, one game", "metrica", "metrica_one_game"), ("Metrica, control off", "metrica", "metrica_control_off"))


def stats(g):
    ok, why = V.verdict(g)
    return dict(mean_gain=float(np.mean(g)), up=sum(x > 0 for x in g), down=sum(x < 0 for x in g),
                tied=sum(x == 0 for x in g), min_gain=float(min(g)), passed=bool(ok), why=why)


def summarize(jobs):
    out = {}
    for b, kind, c in BENCHES + SIDE:
        rs = [j["res"][c] for j in jobs if j["kind"] == kind]
        js = [j for j in jobs if j["kind"] == kind]
        out[b] = {}
        for cand in CANDS:
            gain = lambda k="spell_f1_mean": [x[cand][k] - x["ref"][k] for x in rs]
            out[b][cand] = dict(n=len(rs), ref=float(np.mean([x["ref"]["spell_f1_mean"] for x in rs])), **stats(gain()),
                                macro_gain=float(np.mean(gain("macro_f1"))), trans_gain=float(np.mean(gain("trans_f1"))),
                                ref_spells=float(np.mean([x["ref"]["n_pred_spells"] for x in rs])),
                                new_spells=float(np.mean([x[cand]["n_pred_spells"] for x in rs])),
                                true_spells=float(np.mean([x["ref"]["n_gt_spells"] for x in rs])))
        out[b]["flipped_share"] = float(sum(j["flips"][c][0] for j in js) / max(sum(j["flips"][c][1] for j in js), 1))
        out[b]["log_loss"] = float(np.mean([j["clf"][c]["log_loss"] for j in js]))
        out[b]["auc"] = float(np.mean([j["clf"][c]["auc"] for j in js]))
    passed = {cand: all(out[b][cand]["passed"] for b, _, _ in BENCHES) for cand in CANDS}
    lowest = {cand: float(min(out[b][cand]["min_gain"] for b, _, _ in BENCHES)) for cand in CANDS}
    kept = [c for c in CANDS if passed[c]]
    out["decision"] = dict(passed=passed, lowest_gain=lowest,
                           kept=max(kept, key=lambda c: lowest[c]) if kept else None)
    return out


def print_report(jobs, summary):
    print("\nGain over the reference (spell_f1), weight {} for added".format(WEIGHT))
    print("  {:20s} {:6s} {:>4s} {:>8s} {:>8s} {:>3s} {:>4s} {:>8s} {:>8s} {:>8s} {:>8s} {:>8s}  {}".format(
        "bench", "cand", "runs", "ref", "mean", "up", "down", "min", "macro", "trans", "ref_spl", "new_spl", "verdict"))
    for b, _, _ in BENCHES + SIDE:
        for cand in CANDS:
            s = summary[b][cand]
            print("  {:20s} {:6s} {:4d} {:8.4f} {:+8.4f} {:3d} {:4d} {:+8.4f} {:+8.4f} {:+8.4f} {:8.1f} {:8.1f}  {}".format(
                b, cand, s["n"], s["ref"], s["mean_gain"], s["up"], s["down"], s["min_gain"], s["macro_gain"],
                s["trans_gain"], s["ref_spells"], s["new_spells"],
                ("PASS" if s["passed"] else "FAIL " + s["why"]) if b in [x[0] for x in BENCHES] else "reported only"))
        print("  {:20s} vote flips {:.1%} of spells, classifier log loss {:.4f}, AUC {:.4f} (against this row's labels)".format(
            "", summary[b]["flipped_share"], summary[b]["log_loss"], summary[b]["auc"]))
    d = summary["decision"]
    print("\nPassed all three: added {}, vote {}; lowest gain over the 48 runs: added {:+.4f}, vote {:+.4f}; kept: {}".format(
        d["passed"]["added"], d["passed"]["vote"], d["lowest_gain"]["added"], d["lowest_gain"]["vote"], d["kept"]))
    print("\nGain per run")
    for j in sorted(jobs, key=lambda j: (j["kind"], j["name"], j["seed"])):
        print("  {:14s} seed {}  ".format(j["name"], j["seed"]) + "  ".join(
            "{} ref {:.4f} added {:+.4f} vote {:+.4f}".format(
                c, r["ref"]["spell_f1_mean"], r["added"]["spell_f1_mean"] - r["ref"]["spell_f1_mean"],
                r["vote"]["spell_f1_mean"] - r["ref"]["spell_f1_mean"])
            for c, r in j["res"].items()))


def features_json():
    """team_evidence_features.json's jobs by (match, seed); raises unless it holds the six Metrica runs."""
    saved = {(j["name"], j["seed"]): j for j in json.load(open(FEATURES_JSON))["jobs"]}
    want = {(g, s) for g in METRICA for s in SEEDS}
    assert set(saved) == want, sorted(set(saved) ^ want)
    for j in saved.values():
        assert "spell_f1" in j["res"]["ref"] and "spell_f1" in j["res"]["all5"], (j["name"], j["seed"])
    return saved


def reproduce(pool, saved):
    """Old bench: the one-game "added" scores must equal the "all5" column run by run, and the reference
    must be 0.6685. Raises otherwise."""
    got = {(r["name"], r["seed"]): r for r in pool.starmap(old_metrica_job, [(g, s) for g in METRICA for s in SEEDS])}
    worst = 0.0
    for k, r in sorted(got.items()):
        for mine, theirs in ((r["added"], saved[k]["res"]["all5"]["spell_f1"]), (r["ref"], saved[k]["res"]["ref"]["spell_f1"])):
            worst = max(worst, abs(mine - theirs))
            if abs(mine - theirs) > 1e-9:
                raise RuntimeError("old bench not reproduced for {}: {} against {}".format(k, mine, theirs))
    ref = float(np.mean([r["ref"] for r in got.values()]))
    if round(ref, 4) != REPRODUCTION:
        raise RuntimeError("old bench reference {:.6f}, expected {}".format(ref, REPRODUCTION))
    print("reproduction (old bench): one-game added equals all5 on all 6 runs (largest difference {:.1e}), "
          "reference {:.4f}".format(worst, ref))
    return dict(max_abs_diff=worst, reference=ref)


def reference_check(ref_by_control):
    """ref_by_control: {"metrica": [spellF1 per run], "1.5": [...], "2.5": [...]}; raises unless each mean
    rounds to REFERENCE (abs tol 1e-4)."""
    for c, want in REFERENCE.items():
        got = float(np.mean(ref_by_control[c]))
        if abs(got - want) > 1e-4:
            raise RuntimeError("reference {} is {:.6f}, expected {}".format(c, got, want))
        print("reference {:8s} {:.6f}  ({} runs, expected {})".format(c, got, len(ref_by_control[c]), want))


def train_dead(todo):
    """The current bench's dead ball classifiers (train_spread, no own gaps in training), in S.SPREAD."""
    keys = sorted({(METRICA if n in DFL else G.pool_of(n), s) for n, s in todo}, key=str)
    G.set_share(0.0)
    with mp.get_context("fork").Pool(S.WORKERS) as pool:
        for k, m in pool.imap_unordered(S._spread, keys):
            S.SPREAD[k] = m


def check():
    """Guards, dead ball classifiers, the reference on all 27 runs, the three references, the control
    length and the saved run keys. No team classifier, no candidate."""
    assert external.CONTROL_SECONDS == 1.5, external.CONTROL_SECONDS
    saved = features_json()
    t_start = time.time()
    todo = G.runs()
    assert len(todo) == 27
    train_dead(todo)
    print("[{:5.0f}s] dead ball classifiers trained".format(time.time() - t_start), flush=True)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        got = pool.starmap(reference_job, todo)
    by = {"metrica": [], "1.5": [], "2.5": []}
    for r in got:
        for c, v in r["ref"].items():
            by[c].append(v)
    assert {c: len(v) for c, v in by.items()} == {"metrica": 6, "1.5": 21, "2.5": 21}
    reference_check(by)
    print("control {} s; team_evidence_features.json: {} runs {}; its reference mean {:.4f}".format(
        external.CONTROL_SECONDS, len(saved), sorted(saved), np.mean([j["res"]["ref"]["spell_f1"] for j in saved.values()])))
    print("check passed, no candidate scored, wall time {:.0f}s".format(time.time() - t_start))


def main():
    N.install()
    V.lock_holdout()
    if sys.argv[1] == "check":
        return check()
    out_path = sys.argv[1]
    assert external.CONTROL_SECONDS == 1.5, external.CONTROL_SECONDS
    saved = features_json()
    t_start = time.time()
    single = len(sys.argv) > 2
    seeds = (int(sys.argv[3]),) if single else SEEDS
    games = list(METRICA) + list(DFL)
    G.set_share(0.0)
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for key, X, y in pool.imap_unordered(rows, [(g, s) for g in games for s in seeds]):
            ROWS[key] = X, y
    print("[{:5.0f}s] training rows ready".format(time.time() - t_start), flush=True)
    if not single:                       # the old bench, for the reproduction only
        G.set_share(0.0)
        for s in SEEDS:
            for g in METRICA:
                V._dead_models(tuple(x for x in METRICA if x != g), s, False)
        with mp.get_context("fork").Pool(WORKERS) as pool:
            repro = reproduce(pool, saved)
        print("[{:5.0f}s] reproduction passed".format(time.time() - t_start), flush=True)
    todo = [(sys.argv[2], int(sys.argv[3]))] if single else [(g, s) for g in games for s in SEEDS]
    train_dead(todo)                     # current bench dead ball classifiers, inherited by the workers
    print("[{:5.0f}s] dead ball classifiers trained".format(time.time() - t_start), flush=True)
    if single:
        job = run(todo[0])
        json.dump(job, open(out_path, "w"), indent=1)
        print(json.dumps(job, indent=1))
        return
    jobs = []
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for job in pool.imap_unordered(run, todo):
            jobs.append(job)
            json.dump(dict(jobs=jobs), open(out_path, "w"))
            print("[{:5.0f}s] {}/{} {} seed {} ({:.0f}s)".format(
                time.time() - t_start, len(jobs), len(todo), job["name"], job["seed"], job["job_s"]), flush=True)
    reference_check({c: [j["res"][c]["ref"]["spell_f1_mean"] for j in jobs if j["kind"] == k]
                     for c, k in (("metrica", "metrica"), ("1.5", "dfl"), ("2.5", "dfl"))})
    summary = summarize(jobs)
    json.dump(dict(jobs=jobs, summary=summary, reproduction=repro, wall_s=time.time() - t_start), open(out_path, "w"), indent=1)
    print_report(jobs, summary)
    print("\nwall time {:.0f}s".format(time.time() - t_start))


if __name__ == "__main__":
    main()
