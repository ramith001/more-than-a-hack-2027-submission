"""Score a version of the model on the external validation matches with the official scorer.

    python possession/validate.py              # current model.PARAMS on the tuning matches
    python possession/validate.py --holdout    # also the locked match (only before a submission)
    python possession/validate.py --cv         # leave-one-match-out check of the tuning procedure
    python possession/validate.py --clean-dead     # TUNE only, dead ball trained on the other tuning match
    python possession/validate.py --cv-tune-only   # clean leave-one-out of the tuning procedure, TUNE only
    python possession/validate.py --cv-grid ARG    # same, with the grid in a JSON file or string (not with --clean-dead)
    python possession/validate.py --boot N         # paired bootstrap resamples for the new flags (default 2000)
    python possession/validate.py --scoreboard     # v0 vs current (in-sample), then the clean leave-one-out

The numbers from the five new flags are not comparable with the default run: the dead ball
classifier there is trained on the other tuning match only, not on every other match.

Overfitting guard
-----------------
Tune only on TUNE. HOLDOUT is locked: look at it only to check a version before submitting, never
to choose between settings. If a change helps on TUNE but not on HOLDOUT, it is overfit.
"""

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from possession_metrics import (  # noqa: E402
    BOOTSTRAP_SEED, SPELL_IOUS, evaluate, fill_abstentions, label_codes, matched_gt_flags,
    possession_spells, spell_overlaps, valid_runs)
from possession.external import PITCH, load_game  # noqa: E402
from possession import model  # noqa: E402

TUNE = ("Sample_Game_1", "Sample_Game_2")
HOLDOUT = ("Sample_Game_3",)

KEYS = ("spell_f1_mean", "macro_f1", "n_pred_spells", "n_gt_spells", "alive_dead_f1",
        "team_acc_both_alive")


SEEDS = (0, 1, 2)

_CACHE = {}
_CODES = {}     # the predicted label codes behind each _CACHE entry, for the paired bootstrap
_DEAD = {}
_INPUTS = {}    # a match's tracks with its dead ball probability, the same for every setting

ALL_GAMES = TUNE + HOLDOUT


def _dead_models(train_games, seed=0, gap=False):
    """Dead ball classifiers trained on the given validation matches."""
    from possession import deadball
    key = (tuple(train_games), seed, gap)
    if key not in _DEAD:
        _DEAD[key] = deadball.train(list(train_games), seed=seed, gap=gap)
    return _DEAD[key]


def dead_models_without(name, seed=0, gap=False):
    """Dead ball classifiers trained on every validation match except `name`, so a match is
    never scored by a classifier that has seen it."""
    return _dead_models([g for g in ALL_GAMES if g != name], seed, gap)


def _score_one(predict, params, name, seed, dead_pool=None):
    key = (predict, name, seed, tuple(sorted(params.items())))
    if dead_pool is not None:
        dead_pool = tuple(dead_pool)
        key += (dead_pool,)
    if key not in _CACHE:
        learned = bool(params.get("learned_dead"))
        gap = params.get("dead_gap_features") or False
        ikey = (name, seed, dead_pool, learned, gap)
        if dead_pool is None or ikey not in _INPUTS:
            t, labels, blocks = load_game(name, seed=seed)
            if learned:
                from possession import deadball
                models = dead_models_without(name, seed, gap) if dead_pool is None \
                    else _dead_models(dead_pool, seed, gap)
                t = dict(t, p_dead=deadball.p_dead(models, t, PITCH))
            if dead_pool is not None:
                _INPUTS[ikey] = t, labels, blocks
        else:
            t, labels, blocks = _INPUTS[ikey]
        if params.get("use_restarts"):
            from possession import restarts
            from possession.external import events_for
            tags = restarts.simulated_tags(events_for(name), seed=seed)
            t = dict(t, restarts=restarts.restart_events(
                t, tags, pitch=PITCH if params.get("throw_ins") else None,
                moving=params.get("restart_moving", 6.0)))
        pred = predict(t, PITCH, blocks, len(labels), params)
        _CODES[key] = label_codes(pred)
        r = evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)
        _CACHE[key] = [r[k] for k in KEYS]
    return _CACHE[key]


def score(predict=None, params=None, seeds=(0,), verbose=True, games=TUNE, clean=False):
    predict = predict or model.predict
    params = params or model.PARAMS
    rows = []
    for name in games:
        for seed in seeds:
            pool = tuple(g for g in TUNE if g != name) if clean else None
            rows.append(_score_one(predict, params, name, seed, pool))
            if verbose:
                print("{:14s} seed {}  spellF1 {:.4f}  macroF1 {:.4f}  poss {:4d} / {:4d}  "
                      "alive/dead F1 {:.3f}  team acc {:.3f}".format(name, seed, *rows[-1]))
    mean = np.mean(rows, axis=0)
    if verbose:
        print("{:14s}         spellF1 {:.4f}  macroF1 {:.4f}  poss {:4.0f} / {:4.0f}  "
              "alive/dead F1 {:.3f}  team acc {:.3f}".format("MEAN", *mean))
    return dict(zip(KEYS, mean))


def grid(base, options):
    """All combinations of `options` (dict of name -> list of values) applied on top of `base`."""
    import itertools
    names = list(options)
    for values in itertools.product(*(options[k] for k in names)):
        yield dict(base, **dict(zip(names, values)))


def cross_validate(base, options, reference, games=TUNE + HOLDOUT):
    """Leave one match out: pick the best settings on the others, score them on the one left out.

    This checks the tuning procedure itself. If the settings chosen without a match still beat
    `reference` on that match, the gain generalises; if not, the tuning is overfitting.
    """
    gains = []
    for left_out in games:
        train = tuple(g for g in games if g != left_out)
        best = max(grid(base, options),
                   key=lambda p: score(params=p, games=train, verbose=False)["spell_f1_mean"])
        new = score(params=best, games=(left_out,), verbose=False)["spell_f1_mean"]
        ref = score(params=reference, games=(left_out,), verbose=False)["spell_f1_mean"]
        chosen = {k: best[k] for k in options}
        print("left out {}: chosen {}  -> {:.4f} vs reference {:.4f} ({:+.4f})".format(
            left_out, chosen, new, ref, new - ref), flush=True)
        gains.append(new - ref)
    print("mean out-of-sample gain {:+.4f}".format(np.mean(gains)))
    return gains


V0 = dict(model.PARAMS, switch_team=12.0, switch_touch=0, touch_weight=0, flight_speed=0,
          touch_direction=False, touch_angle=0, touch_dspeed=0, learned_dead=False,
          min_dead=25, scuffle_radius=0, touch_reach=2.5)
TOUCH_GRID = {"switch_touch": [6, 10, 14], "touch_jerk": [1.0, 1.5, 2.5],
              "touch_reach": [2.0, 2.5, 3.0]}


# -- clean tuning protocol: TUNE only, the dead ball classifier never sees the match it scores ----

def lock_holdout():
    """Make loading a HOLDOUT match an error, in this module and in possession.external (which
    deadball.train imports load_game from when it runs)."""
    from possession import external
    real = external.load_game

    def locked(name, seed=0, **kw):
        if name in HOLDOUT:
            raise RuntimeError("{} is the locked holdout".format(name))
        return real(name, seed=seed, **kw)
    external.load_game = locked
    globals()["load_game"] = locked


def _clean_score(params, name, seed, pool=None):
    pool = tuple(g for g in TUNE if g != name) if pool is None else tuple(pool)
    return _score_one(model.predict, params, name, seed, pool)[0]


def _clean_codes(params, name, seed):
    _clean_score(params, name, seed)
    key = (model.predict, name, seed, tuple(sorted(params.items())),
           tuple(g for g in TUNE if g != name))
    return _CODES[key]


def loo_tune(base, options, reference, games=TUNE, seeds=SEEDS):
    """Leave one match out inside `games`: pick on the others by mean spellF1 over the seeds,
    score on the one left out. Ties go to the first setting in grid() order.

    Each training match is scored with a dead ball classifier trained on the training matches
    themselves, so the left out match never affects the choice. The left out match is scored with
    the classifier trained on the training matches.

    One row per left out match and seed: chosen settings, new, reference, gain."""
    rows = []
    for left_out in games:
        train = [g for g in games if g != left_out]
        best = max(grid(base, options),
                   key=lambda p: np.mean([_clean_score(p, g, s, train) for g in train for s in seeds]))
        for seed in seeds:
            new, ref = _clean_score(best, left_out, seed), _clean_score(reference, left_out, seed)
            rows.append({"left_out": left_out, "seed": seed, "params": best,
                         "chosen": {k: best[k] for k in options}, "new": new, "ref": ref,
                         "gain": new - ref})
    return rows


def verdict(gains):
    """(ok, reason): every gain above zero and the mean at least +0.005."""
    bad = [g for g in gains if not g > 0]
    reasons = []
    if bad:
        reasons.append("{} of {} gains not above 0".format(len(bad), len(gains)))
    if np.mean(gains) < 0.005:
        reasons.append("mean gain {:+.4f} below +0.005".format(np.mean(gains)))
    return not reasons, "; ".join(reasons)


def paired_boot(name, preds_new_by_seed, preds_ref_by_seed, n):
    """Paired bootstrap of the spellF1 gain on one match, resampling its possessions.

    Both versions and all seeds see the same resample. Same scoring as possession_metrics
    (greedy match per IoU, abstentions filled, n_pred fixed, fp clamped at 0).
    Returns (gain, low, high, P(gain <= 0)); the last three are None when n is 0."""
    gts = [label_codes(load_game(name, seed=s)[1]) for s in preds_new_by_seed]
    assert all(np.array_equal(gts[0], g) for g in gts), "ground truth depends on the seed"
    runs = valid_runs(gts[0])
    gt_spells = possession_spells(gts[0], runs)
    n_gt = len(gt_spells)

    def flags(preds):
        out = []
        for p in preds.values():
            spells = possession_spells(fill_abstentions(p, runs), runs)
            overlaps = spell_overlaps(gt_spells, spells)
            out.append((np.vstack([matched_gt_flags(overlaps, n_gt, th) for th in SPELL_IOUS])
                        .astype(np.int64), len(spells)))
        return out

    def f1(tp, n_pred):
        denominator = 2 * tp + np.maximum(0, n_pred - tp) + n_gt - tp
        return np.mean(np.where(tp > 0, 2 * tp / np.maximum(denominator, 1), 0.0), axis=-1)

    new, ref = flags(preds_new_by_seed), flags(preds_ref_by_seed)
    gain = np.mean([f1(F.sum(1), k) for F, k in new]) - np.mean([f1(F.sum(1), k) for F, k in ref])
    if not n:
        return gain, None, None, None
    rng = np.random.RandomState(BOOTSTRAP_SEED)
    counts = np.vstack([np.bincount(rng.randint(0, n_gt, n_gt), minlength=n_gt)
                        for _ in range(n)]).astype(np.int64)
    draws = (np.mean([f1(counts @ F.T, k) for F, k in new], axis=0)
             - np.mean([f1(counts @ F.T, k) for F, k in ref], axis=0))
    low, high = np.percentile(draws, (2.5, 97.5))
    return gain, float(low), float(high), float(np.mean(draws <= 0))


def _boot(name, new_params, ref_params, seeds, n):
    gain, low, high, p = paired_boot(
        name, {s: _clean_codes(new_params, name, s) for s in seeds},
        {s: _clean_codes(ref_params, name, s) for s in seeds}, n)
    point = np.mean([_clean_score(new_params, name, s) - _clean_score(ref_params, name, s)
                     for s in seeds])
    assert abs(gain - point) < 1e-9, "bootstrap point gain {} != scored gain {}".format(gain, point)
    return gain, low, high, p


def _boot_text(gain, low, high, p):
    if low is None:
        return "mean gain {:+.4f}".format(gain)
    return "mean gain {:+.4f}  95% CI [{:.4f}, {:.4f}]  P(gain<=0) {:.3f}".format(gain, low, high, p)


def report_loo(base, options, reference, boot_n, seeds=SEEDS):
    """Print the leave-one-out table, a bootstrap line per match and the verdict; True on PASS."""
    rows = loo_tune(base, options, reference, seeds=seeds)
    print("{:14s} {:4s}  {:34s} {:>6s} {:>6s} {:>7s}".format(
        "left out", "seed", "chosen", "new", "ref", "gain"))
    for r in rows:
        print("{:14s} {:4d}  {:34s} {:.4f} {:.4f} {:+.4f}".format(
            r["left_out"], r["seed"], str(r["chosen"]), r["new"], r["ref"], r["gain"]))
    for name in dict.fromkeys(r["left_out"] for r in rows):
        mine = [r for r in rows if r["left_out"] == name]
        print("{:14s} {}".format(name, _boot_text(*_boot(
            name, mine[0]["params"], reference, seeds, boot_n))))
    gains = [r["gain"] for r in rows]
    ok, why = verdict(gains)
    print("mean out-of-sample gain {:+.4f}".format(np.mean(gains)))
    print("VERDICT PASS" if ok else "VERDICT FAIL: " + why)
    return ok


def scoreboard(base, options, reference, boot_n):
    print("in-sample: current settings were tuned on these matches (not evidence for a change)")
    print("{:14s} {:4s}  {:>6s} {:>8s} {:>7s}".format("game", "seed", "v0", "current", "gain"))
    for name in TUNE:
        for seed in SEEDS:
            v0, cur = _clean_score(V0, name, seed), _clean_score(model.PARAMS, name, seed)
            print("{:14s} {:4d}  {:.4f} {:8.4f} {:+.4f}".format(name, seed, v0, cur, cur - v0))
    for name in TUNE:
        print("{:14s} {}".format(name, _boot_text(*_boot(name, model.PARAMS, V0, SEEDS, boot_n))))
    print("clean leave-one-out (out-of-sample)")
    return report_loo(base, options, reference, boot_n)


NEW_FLAGS = ("--clean-dead", "--cv-tune-only", "--cv-grid", "--boot", "--scoreboard")


def _fail(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def _read_grid_spec(arg):
    """ARG is a JSON string if it starts with "{", else the path of a JSON file. The spec is
    {"base": {...}, "grid": {name: [values]}, "reference": {...}}; each part is laid over
    model.PARAMS, and an omitted "reference" means model.PARAMS."""
    if arg.lstrip().startswith("{"):
        text = arg
    elif os.path.isfile(arg):
        text = open(arg).read()
    else:
        _fail("--cv-grid: file not found: {}".format(arg))
    try:
        spec = json.loads(text)
    except ValueError as e:
        _fail("--cv-grid: cannot read {!r}: {}".format(arg, e))
    if not isinstance(spec, dict):
        _fail("--cv-grid: the top level must be an object")
    for part in ("base", "grid", "reference"):
        if not isinstance(spec.get(part, {}), dict):
            _fail("--cv-grid: {} must be an object".format(part))
    if not all(isinstance(v, list) and v for v in spec.get("grid", {}).values()):
        _fail("--cv-grid: every grid entry must be a non-empty list of values")
    if any(g in text for g in HOLDOUT):
        _fail("--cv-grid: the holdout is locked")
    for part in ("base", "grid", "reference"):
        if "seed" in spec.get(part, {}):
            _fail("--cv-grid: seed is not a setting")
        for k in spec.get(part, {}):
            if k not in model.PARAMS:
                print("warning: {!r} in {} is not a model.PARAMS key".format(k, part),
                      file=sys.stderr)
    return (dict(model.PARAMS, **spec.get("base", {})), spec.get("grid", {}),
            dict(model.PARAMS, **spec.get("reference", {})))


def _flag_value(flag):
    i = sys.argv.index(flag)
    if i + 1 >= len(sys.argv):
        _fail("{} needs a value".format(flag))
    return sys.argv[i + 1]


if __name__ == "__main__":
    if any(f in sys.argv for f in NEW_FLAGS) and ("--cv" in sys.argv or "--holdout" in sys.argv):
        _fail("{} cannot be combined with --cv or --holdout".format(" / ".join(NEW_FLAGS)))
    if any(f in sys.argv for f in NEW_FLAGS):
        try:
            boot_n = int(_flag_value("--boot")) if "--boot" in sys.argv else 2000
        except ValueError:
            _fail("--boot needs a whole number")
        if boot_n < 0:
            _fail("--boot needs a whole number of at least 0")
        if "--clean-dead" in sys.argv and "--cv-grid" in sys.argv:
            _fail("--clean-dead cannot be combined with --cv-grid")
        base, options, reference = model.PARAMS, TOUCH_GRID, V0
        if "--cv-grid" in sys.argv:
            base, options, reference = _read_grid_spec(_flag_value("--cv-grid"))
        lock_holdout()
        if "--scoreboard" in sys.argv:
            sys.exit(0 if scoreboard(base, options, reference, boot_n) else 1)
        elif "--cv-tune-only" in sys.argv or "--cv-grid" in sys.argv:
            sys.exit(0 if report_loo(base, options, reference, boot_n) else 1)
        else:
            print("clean-dead (TUNE only, dead ball trained on the other tuning match)")
            score(seeds=SEEDS, clean=True)
    elif "--cv" in sys.argv:
        cross_validate(model.PARAMS, {"switch_touch": [6, 10, 14], "touch_jerk": [1.0, 1.5, 2.5],
                                      "touch_reach": [2.0, 2.5, 3.0]}, V0)
    else:
        print("tuning matches")
        score()
        if "--holdout" in sys.argv:
            print("locked holdout match")
            score(games=HOLDOUT)
