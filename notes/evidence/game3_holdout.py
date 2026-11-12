"""One look at the locked holdout match (Metrica Game 3) with the hand-set model (without the learned team probability). Run it once.

The dead ball classifiers are trained as in the submission (deadball.train_spread, Metrica Games 1 and 2
under random degradations). Game 3 is scored under random degradations drawn from the same spread, with
streams and degrade seeds that training never uses, seeds 0-2 and 5 draws each (15 runs). The same runs
are scored for Games 1 and 2, each with the classifier trained on the other game only, so the numbers
can be compared like for like. Nothing from the challenge match is read. This is a check of the hand-set
model, not a way to choose settings: nothing may be changed because of what it shows.

    python notes/evidence/game3_holdout.py                      # check run on Games 1 and 2 only
    MTAH_GAME3_LOOK=1 python notes/evidence/game3_holdout.py --game3   # the single look

The Game 3 run refuses to start twice: it writes notes/evidence/game3_holdout_result.json and stops
if that file exists.
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import numpy as np  # noqa: E402

from possession import deadball, external, model, restarts  # noqa: E402
from possession.external import PITCH  # noqa: E402
from possession_metrics import evaluate, label_codes  # noqa: E402

TUNE = ("Sample_Game_1", "Sample_Game_2")
HOLDOUT = "Sample_Game_3"
INDEX = {"Sample_Game_1": 20, "Sample_Game_2": 21, "Sample_Game_3": 22}
SEEDS = (0, 1, 2)
DRAWS = 5
RESULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "game3_holdout_result.json")


def score_game(name, pool):
    out = []
    for seed in SEEDS:
        models = deadball.train_spread(list(pool), seed=seed)
        clean, raw, blocks = external._clean_game(name)
        labels = external.require_control(raw, external.bench_events(name, clean["bx"]),
                                          external.CONTROL_SECONDS)
        events = external.events_for(name)
        for draw in range(DRAWS):
            setting = deadball.spread_setting(np.random.default_rng([seed, INDEX[name], draw, 90]))
            dseed = 50_000_000 + seed * 10_000 + INDEX[name] * 100 + draw
            t, _ = external.load_spread(name, dseed, setting)
            t = dict(t, p_dead=deadball.p_dead(models, t, PITCH))
            tags = restarts.simulated_tags(events, seed=seed)
            t = dict(t, restarts=restarts.restart_events(
                t, tags, pitch=PITCH if model.PARAMS.get("throw_ins") else None,
                moving=model.PARAMS.get("restart_moving", 6.0)))
            pred = model.predict(t, PITCH, blocks, len(labels), model.PARAMS)
            out.append(evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)["spell_f1_mean"])
        print(name, "seed", seed, "mean so far %.4f" % np.mean(out), flush=True)
    return out


def main():
    look = "--game3" in sys.argv
    if look:
        if os.environ.get("MTAH_GAME3_LOOK") != "1":
            sys.exit("set MTAH_GAME3_LOOK=1 to take the single look at Game 3")
        if os.path.exists(RESULT):
            sys.exit("Game 3 has already been looked at: see " + RESULT)
    res = {}
    for name in TUNE:
        res[name] = score_game(name, [g for g in TUNE if g != name])
    if look:
        res[HOLDOUT] = score_game(HOLDOUT, TUNE)
    for name, r in res.items():
        print("%s: mean %.4f, sd over runs %.4f, runs %d" % (name, np.mean(r), np.std(r), len(r)))
    if look:
        tune = [x for g in TUNE for x in res[g]]
        print("Games 1-2 pooled %.4f, Game 3 %.4f, difference %+.4f" % (
            np.mean(tune), np.mean(res[HOLDOUT]), np.mean(res[HOLDOUT]) - np.mean(tune)))
        json.dump(res, open(RESULT, "w"))


if __name__ == "__main__":
    main()
