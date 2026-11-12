"""Learned team evidence: P(home has the ball) per frame, added to the model's emissions.

A gradient boosted classifier is trained on external matches only, Metrica Sample_Game_1 and
Sample_Game_2 and the seven DFL open matches, never Sample_Game_3 and never our match. Training rows
are every 5th live frame with the ball seen, from tracks degraded as on the validation bench (seed 0,
no gaps copied from our match), labelled H or A with the 1.5 s control rule. The features are ball and
player measures: distances to each team, speeds, time since each team's last touch, the touch owner
(feats), and five groups (extra): player density around the ball, the nearest players' movement
relative to the ball, the relative distance advantage, the ball's recent path, and rolling pressure.
Everything looks at tracking only; the rolling and path features look back only.

The model adds WEIGHT * log P(home) to H and WEIGHT * log P(away) to A (model.emissions, team_weight).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from possession import external, external_dfl, model

P = dict(model.PARAMS)
FPS = 25
WEIGHT = 0.25
SEED = 0
STEP = 5
GROUPS = ("density", "alignment", "advantage", "trajectory", "pressure")
METRICA = ("Sample_Game_1", "Sample_Game_2")
POOL = METRICA + external_dfl.GAMES


def feats(t, p=P):
    bx, by = model.fill_ball(t["bx"], t["by"], p["max_gap"])
    n = len(bx); ok = np.isfinite(bx)
    d, tf, dh, da = model.team_distances(t, bx, by)
    speed, jerk = model.ball_kinematics(bx, by)
    f = [dh, da, np.minimum(dh, da), da - dh, speed, jerk, ok.astype(float)]
    for team in (0, 1):
        dd = np.where(tf == team, d, np.inf)
        s = np.sort(dd, axis=1)
        f += [s[:, 1], (dd < 3).sum(1), (dd < 6).sum(1)]
        j = np.argmin(dd, axis=1)
        sp = np.hypot(np.gradient(np.where(np.isfinite(t["px"]), t["px"], np.nan), axis=0),
                      np.gradient(np.where(np.isfinite(t["py"]), t["py"], np.nan), axis=0))[np.arange(n), j] * 25
        f.append(sp)
    frames, teams, _ = model.detect_touches(t, bx, by, p)
    for team in (0, 1):
        last = np.full(n, 10 ** 4.0); fr = frames[teams == team]
        idx = np.searchsorted(fr, np.arange(n), side="right") - 1
        ok2 = idx >= 0
        last[ok2] = np.arange(n)[ok2] - fr[idx[ok2]]
        f.append(np.minimum(last, 750))
    owner = model.touch_owner(frames, teams, n, p)
    f.append(owner.astype(float))
    return np.column_stack(f).astype(np.float32), ok


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


def features(t, blocks):
    """(n, k) features in a fixed order, and the frames with the ball seen."""
    F, ok = feats(t)
    E = extra(t, blocks)
    return np.column_stack([F] + [E[g] for g in GROUPS]), ok


def training_rows(name, seed=SEED):
    """Features and H labels (1 home, 0 away) of one external match."""
    assert name != "Sample_Game_3"
    assert external.CONTROL_SECONDS == 1.5, external.CONTROL_SECONDS
    if name in external_dfl.GAMES:
        t, y, blocks = external_dfl.load_game(name, seed=seed, control=1.5, own_gap_share=0.0)
    else:
        t, y, blocks = external.load_game(name, seed=seed, own_gap_share=0.0)
    F, ok = features(t, blocks)
    idx = np.flatnonzero(np.isin(y, ["H", "A"]) & ok)[::STEP]
    return F[idx], (np.asarray(y)[idx] == "H").astype(int)


def train(pool=POOL, seed=SEED):
    rows = [training_rows(g, seed) for g in pool]
    X = np.concatenate([r[0] for r in rows])
    y = np.concatenate([r[1] for r in rows])
    clf = HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05, random_state=seed)
    return clf.fit(X, y)


def p_home(clf, t, blocks):
    """P(home has the ball) per frame and the frames with the ball seen, for t["p_team"], t["p_ok"]."""
    F, ok = features(t, blocks)
    return clf.predict_proba(F)[:, 1], ok
