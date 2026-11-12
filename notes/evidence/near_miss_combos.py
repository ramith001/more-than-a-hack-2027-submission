"""Pre-registered near-miss combinations on the Metrica bench (Games 1-2, seeds 0-2, clean leave-one-out).

switch_control 5 or 10, alone and with dead_gap_features ball or all, each against the current defaults.
Run from the repo root: python notes/evidence/near_miss_combos.py. Test only.
"""
import sys, os, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.getcwd()); sys.path.insert(0, os.path.join(os.getcwd(), "evaluation"))
import numpy as np
from possession import validate as V, model
V.lock_holdout()
base = dict(model.PARAMS)
ref = [V._clean_score(base, g, s) for g in V.TUNE for s in (0,1,2)]
print("base mean %.4f" % np.mean(ref), flush=True)
for sc in (5, 10):
    for gap in (False, "ball", "all"):
        p = dict(base, switch_control=sc, dead_gap_features=gap)
        r = [V._clean_score(p, g, s) for g in V.TUNE for s in (0,1,2)]
        gains = [a-b for a,b in zip(r, ref)]
        ok, why = V.verdict(gains)
        print("sc=%s gap=%s mean gain %+.4f  runs up %d/6  %s" % (sc, gap, np.mean(gains), sum(g>0 for g in gains), "PASS" if ok else why), flush=True)
