import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from possession.io import load_tracking, write_submission, forced_dead, forced_alive, clean_keepers, PITCH  # noqa: E402
from possession.model import predict, PARAMS  # noqa: E402

if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "evaluation", "results", "v2.csv")
    t = clean_keepers(load_tracking())
    t["forced_dead"] = forced_dead(t)
    t["forced_alive"] = forced_alive(t, seen_only=PARAMS.get("kickoff_seen_only", False))
    if PARAMS.get("learned_dead"):
        from possession import deadball
        # external data only: Metrica Games 1 and 2 under a spread of degradations, our match not read
        t["p_dead"] = deadball.p_dead(
            deadball.train_spread(("Sample_Game_1", "Sample_Game_2"), gap=PARAMS.get("dead_gap_features") or False),
            t, PITCH)
    if PARAMS.get("use_restarts"):
        from possession import restarts
        t["restarts"] = restarts.restart_events(
            t, restarts.tags_from_smart_tagging(), pitch=PITCH if PARAMS.get("throw_ins") else None,
            moving=PARAMS.get("restart_moving", 6.0))
    blocks = half_video_blocks(GAME)
    p = PARAMS
    if PARAMS.get("team_evidence"):
        from possession import team_evidence
        # external data only: Metrica Games 1 and 2 and the seven DFL open matches, our match not read
        t["p_team"], t["p_ok"] = team_evidence.p_home(team_evidence.train(), t, blocks)
        p = dict(PARAMS, team_weight=team_evidence.WEIGHT)
    labels = predict(t, PITCH, blocks, len(t["bx"]), p)
    write_submission(out, labels)
    print(out, {k: int((labels == k).sum()) for k in "HADX"})
