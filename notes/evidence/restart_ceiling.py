"""Ceiling of the rule "a stoppage lasts until the restart" with perfect restart times.

Uses the logged restarts of the Metrica matches as the detected restarts (lag 0), then with the
detection late by 1 s and 2 s. Games 1-2, seeds 0-2. Run from the repo root:
python notes/evidence/restart_ceiling.py. Test only.
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "evaluation")); sys.path.insert(0, os.path.join(os.getcwd(), "notes", "evidence"))
import numpy as np
import restart_rule_check as R
from possession import validate as V, model
from possession_metrics import evaluate, label_codes
V.lock_holdout()
for lag in (0, 25, 50):
    gains=[]
    for name in V.TUNE:
        true = R.logged_restarts(name)
        for seed in (0,1,2):
            t, labels, pred = R.predictions(name, seed, model.PARAMS)
            found = true + lag
            new = R.apply_rule(pred, found)
            s0 = evaluate(label_codes(labels), label_codes(pred), bootstrap_n=0)["spell_f1_mean"]
            s1 = evaluate(label_codes(labels), label_codes(new), bootstrap_n=0)["spell_f1_mean"]
            gains.append(s1-s0)
    print("oracle restarts, lag %d frames: mean gain %+.4f  up %d/6" % (lag, np.mean(gains), sum(g>0 for g in gains)), flush=True)
