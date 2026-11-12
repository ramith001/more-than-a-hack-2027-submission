"""Restarts from Smart Tagging: who has the ball when play restarts after a set piece.

Smart Tagging marks set pieces (throw-ins, corners, goal kicks, free kicks, kick-offs) with the
team that takes them, except throw-ins, which are tagged N/A. A tag starts inside the stoppage, a
few seconds before the restart. For each tag with a team, the restart is found as the first moment
after the tag start where the ball goes from still to moving fast. Under the Laws of the Game the
team taking a set piece has the ball when it is taken, so for a short moment after the restart the
ball is live and belongs to that team; normal touches take over after that.

The validation matches have no Smart Tagging, so for them tags are simulated from the set piece
events: a tag starting a random 2 to 12 seconds before the restart (Smart Tagging's own median is
about 7 s), throw-ins with no team, and a share of tags dropped, because Smart Tagging is automatic
and misses some.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.model import ball_kinematics  # noqa: E402

H, A = 0, 1


def find_restart(t, start, window=500, still=2.5, moving=6.0, ahead=6):
    """First frame after `start` where the seen ball is still and moving fast `ahead` frames later."""
    speed, _ = ball_kinematics(t["bx"], t["by"])
    end = min(start + window, len(speed) - ahead)
    for f in range(start, end):
        if np.isfinite(t["bx"][f]) and speed[f] < still and speed[f + ahead] > moving:
            return f
    return None


def thrower_team(t, r, pitch, line=2.0, reach=1.5, look=10):
    """Team taking a throw-in at restart frame r, or None if unsure.

    The thrower holds the ball at the touchline, so the player nearest the ball in the moment
    before it is thrown is the thrower. Only used when the ball is at a touchline and a player is
    within `reach` of it (on the validation matches this picks the right team 94 to 100% of the
    time; the last touch before the ball went out is a coin flip with our touch detection)."""
    width = pitch[1]
    by = t["by"][r]
    if not np.isfinite(by) or min(abs(by), abs(by - width)) > line:
        return None
    sl = slice(max(0, r - look), r + 1)
    d = np.hypot(t["px"][sl] - t["bx"][sl, None], t["py"][sl] - t["by"][sl, None])
    if not np.isfinite(d).any():
        return None
    best = np.nanmin(d, axis=0)
    j = int(np.nanargmin(best))
    if best[j] > reach:
        return None
    tf = t.get("team_frame")
    return int(tf[r, j]) if tf is not None else int(t["team"][j])


def restart_events(t, tags, window=500, pitch=None, moving=6.0):
    """[(restart_frame, team)] for set piece tags [(start_frame, team or None)].

    Tags without a team are throw-ins; with `pitch` given, their team comes from thrower_team."""
    out = []
    for start, team in tags:
        if team is None and pitch is None:
            continue
        r = find_restart(t, start, window, moving=moving)
        if r is None:
            continue
        if team is None:
            team = thrower_team(t, r, pitch)
            if team is None:
                continue
        out.append((r, team))
    return out


def tags_from_smart_tagging():
    from possession.io import smart_tagging_instances
    tags = []
    for start, _, code, tg in smart_tagging_instances():
        if code != "SET PIECES" or "kick off" in (tg.get("Type") or "").lower():
            continue
        team = {"Barcelona": H, "Atletic Club": A}.get(tg.get("Team"))
        tags.append((start, team))
    return tags


def simulated_tags(events, seed=0, drop=0.15, lead=(2.0, 12.0), fps=25):
    """Set piece tags as Smart Tagging would give them, built from validation event data."""
    from possession.external import is_restart
    rng = np.random.default_rng(seed)
    tags = []
    for e in events:
        if not is_restart(e) or "KICK OFF" in e["subtype"]:
            continue
        if rng.random() < drop:
            continue
        start = e["start"] - int(rng.uniform(*lead) * fps)
        team = None if "THROW" in e["subtype"] else e["team"]
        tags.append((start, team))
    return tags


def apply(ll, events, p):
    """Add restart evidence to the per-frame log-likelihoods (columns H, A, D)."""
    k = int(p.get("restart_frames", 12))
    w = p.get("restart_weight", 2.0)
    alive = int(p.get("restart_alive", 0))
    before = int(p.get("restart_dead_before", 0))
    for r, team in events:
        ll[r:r + max(k, alive), 2] = -50.0
        ll[r:r + k, team] += w
        if before:
            ll[max(0, r - before):r, :2] = -50.0
    return ll
