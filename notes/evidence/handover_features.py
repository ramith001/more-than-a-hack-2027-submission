"""Do two simple tracking signals tell a real handover from a false one? Measured before building.

Candidates: every detected touch (model.detect_touches on the degraded bench tracking) by a team
other than the one that made the previous detected touch, with the truth live 1 s before and 2 s
after it. The truth decides which are real: real if the true team 1 s before is the other team and
2 s after is the touching team, false if the true team is the other team at both.

Signals, all from the tracking:
  carry       share of the next 25 frames in which the touching player (the touching team's nearest
              player at the touch) is within 2 m of the ball; a deflection is a carry share under
              0.2 with no further touch by that team in the next 25 frames
  next touch  the next detected touch within 3 s: same player, a team-mate (a completed pass),
              the other team (the ball goes back), or none

Also: how many logged touches have no detected touch within 8 frames, how many detected touches
match a logged one and have its team, how late the late handovers of team_ceiling.py are, and
whether a touch by the new team is detected within 1 s of each of them.
Games 1-2, seeds 0-2. Game 3 and our match are not used. Test only.

    python notes/evidence/handover_features.py
"""
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import team_ceiling as T  # noqa: E402
from possession import validate as V, model  # noqa: E402

FPS = 25


def auc(pos, neg):
    """Probability a real candidate scores higher than a false one (ties count half)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.r_[pos, neg]
    ranks = np.argsort(np.argsort(allv, kind="mergesort"), kind="mergesort") + 1.0
    for v in np.unique(allv):
        m = allv == v
        ranks[m] = ranks[m].mean()
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def nearest_slot(t, tf, bx, by, f, team):
    d = np.hypot(t["px"][f] - bx[f], t["py"][f] - by[f])
    d = np.where((tf[f] == team) & np.isfinite(d), d, np.inf)
    return int(np.argmin(d)) if np.isfinite(d).any() else -1


def main():
    V.lock_holdout()
    p = dict(model.PARAMS)
    rows = []          # (real, contested, carry share, deflection, next kind)
    stats = np.zeros(4, int)   # logged, detected, matched, matched right team
    late = []
    missed = 0
    for name in V.TUNE:
        true = T.logged_touches(name)
        for seed in V.SEEDS:
            t, labels, pred = T.E.predict(p, name, seed)
            bx, by = model.fill_ball(t["bx"], t["by"], p["max_gap"])
            frames, teams, contested = model.detect_touches(t, bx, by, p)
            _, tf, _, _ = model.team_distances(t, bx, by)
            stats[0] += len(true)
            stats[1] += len(frames)
            missed += sum(T.nearest(frames, f)[1] > T.MATCH for f, _, _ in true)
            for f, tm in zip(frames, teams):
                j, dist = T.nearest(true[:, 0], f)
                if dist <= T.MATCH:
                    stats[2] += 1
                    stats[3] += int(true[j, 1] == tm)
            # late handovers, as in team_ceiling.fix_late
            for a, b in T.runs_of(labels)[1:]:
                bef, now = labels[a - 1], labels[a]
                if bef in "HA" and now in "HA" and bef != now and pred[a] == bef:
                    k = np.flatnonzero(pred[a:a + 10 * FPS] == now)
                    if len(k) and k[0] > FPS:
                        tm = "HA".index(now)
                        near = np.flatnonzero((frames >= a - T.MATCH) & (frames <= a + FPS) & (teams == tm))
                        kind = "no touch"
                        if len(near):
                            f = frames[near[0]]
                            nxt = np.flatnonzero((frames > f) & (frames <= f + 3 * FPS))
                            kind = "next none" if not len(nxt) else (
                                "next other team" if teams[nxt[0]] != tm else "next own team")
                        late.append((k[0] / FPS, kind))
            n = len(labels)
            for i in range(1, len(frames)):
                f, tm = frames[i], teams[i]
                if tm == teams[i - 1] or f - FPS < 0 or f + 2 * FPS >= n:
                    continue
                before, after = labels[f - FPS], labels[f + 2 * FPS]
                other = "HA"[1 - tm]
                if before != other or after not in "HA":
                    continue
                real = after == "HA"[tm]
                s = nearest_slot(t, tf, bx, by, f, tm)
                if s < 0:
                    continue
                w = np.arange(f + 1, f + FPS + 1)
                d = np.hypot(t["px"][w, s] - bx[w], t["py"][w, s] - by[w])
                carry = float(np.mean(np.where(np.isfinite(d), d, np.inf) <= 2.0))
                nxt = np.flatnonzero((frames > f) & (frames <= f + 3 * FPS))
                own_soon = np.any((frames > f) & (frames <= f + FPS) & (teams == tm))
                if not len(nxt):
                    kind = "none"
                else:
                    g, gt = frames[nxt[0]], teams[nxt[0]]
                    if gt != tm:
                        kind = "other team"
                    elif nearest_slot(t, tf, bx, by, g, tm) == s:
                        kind = "same player"
                    else:
                        kind = "team-mate"
                rows.append((real, bool(contested[i]), carry, carry < 0.2 and not own_soon, kind))
    print("logged touches {}  detected {}  detected matching a logged one {} ({:.0%})  "
          "right team {:.0%} of those".format(stats[0], stats[1], stats[2], stats[2] / stats[1],
                                              stats[3] / max(stats[2], 1)))
    print("logged touches with no detected touch within {} frames: {} ({:.0%})".format(
        T.MATCH, missed, missed / stats[0]))
    secs = np.array([x for x, _ in late])
    print("late handovers (over 1 s, within 10 s): {} in 6 runs, median {:.1f} s, quartiles {:.1f} to {:.1f} s"
          .format(len(secs), np.median(secs), *np.percentile(secs, [25, 75])))
    for kind in ("no touch", "next own team", "next other team", "next none"):
        k = [x for x, kd in late if kd == kind]
        print("  touch by the new team within 1 s of the true change: {:16s} {:4d} ({:.0%})".format(
            kind, len(k), len(k) / len(late)))
    for label, sel in (("all candidates", lambda r: True), ("contested", lambda r: r[1])):
        rr = [r for r in rows if sel(r)]
        real = [r for r in rr if r[0]]
        false = [r for r in rr if not r[0]]
        print("\n{}: {} real, {} false handovers".format(label, len(real), len(false)))
        print("  carry share: median real {:.2f}, false {:.2f}; AUC {:.3f}".format(
            np.median([r[2] for r in real]), np.median([r[2] for r in false]),
            auc([r[2] for r in real], [r[2] for r in false])))
        for name, cond in (("carry share >= 0.6", lambda r: r[2] >= 0.6), ("deflection", lambda r: r[3])):
            nr, nf = sum(cond(r) for r in real), sum(cond(r) for r in false)
            print("  {:20s} real {:4d} ({:.0%})  false {:4d} ({:.0%})  share real when it holds {:.0%}".format(
                name, nr, nr / len(real), nf, nf / len(false), nr / max(nr + nf, 1)))
        for kind in ("same player", "team-mate", "other team", "none"):
            nr, nf = sum(r[4] == kind for r in real), sum(r[4] == kind for r in false)
            print("  next touch {:12s} real {:4d} ({:.0%})  false {:4d} ({:.0%})  share real {:.0%}".format(
                kind, nr, nr / len(real), nf, nf / len(false), nr / max(nr + nf, 1)))
        base = len(real) / len(rr)
        print("  share real overall {:.0%}".format(base))


if __name__ == "__main__":
    main()
