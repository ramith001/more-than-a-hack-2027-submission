"""Dead ball detection, learned from the validation matches.

For every frame a set of signals is computed from the tracking alone (no labels, nothing specific
to one match), and a gradient-boosted classifier trained on matches with known dead/live labels
turns them into the probability that the ball is dead. That probability replaces the hand-written
dead-ball rule in model.emissions.

Signals, each over windows around the frame (the model works on the whole match at once, so it
can look both before and after a frame):
  - is the ball seen around the frame (player model only, see features())
  - how fast the ball moves, and how long since / until it last moved fast
  - is the ball outside the lines, and how far from the nearest line (restart spots)
  - is the ball standing still
  - how fast the players move, and whether they are slowing down or speeding up
  - how many players are tracked, and how close the nearest one is to the ball
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.model import fill_ball, smooth_nan  # noqa: E402

FPS = 25
SHORT_TRENDS = False   # extra player slow-down features over 1 to 3 s (test)
SPREAD_COPIES = 10     # degraded versions of each training match in train_spread


def _window_mean(x, half):
    """Centred mean over +-half frames, ignoring NaN; NaN where the whole window is NaN."""
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    m = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - half), np.minimum(len(x), i + half + 1)
    cnt = m[hi] - m[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _window_max(x, half):
    from scipy.ndimage import maximum_filter1d
    return maximum_filter1d(np.where(np.isfinite(x), x, -1e9), 2 * half + 1, mode="nearest")


def _frames_since(flag, cap=750):
    """Frames since flag was last True (cap if never)."""
    out = np.empty(len(flag))
    last = -cap
    for i, v in enumerate(flag):
        if v:
            last = i
        out[i] = min(i - last, cap)
    return out


def _frames_until(flag, cap=750):
    return _frames_since(flag[::-1], cap)[::-1]


def _last_seen(x, seen):
    """x at the last frame up to now where the ball was seen (NaN before the first)."""
    idx = np.where(seen, np.arange(len(x)), -1)
    idx = np.maximum.accumulate(idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], np.nan)


def _next_seen(x, seen):
    return _last_seen(x[::-1], seen[::-1])[::-1]


def _feature_table(t, pitch, gap=False):
    """Named signals. Ball signals use only frames where the ball is actually seen. With `gap`
    ("ball", "team" or "all"; True is "all"), the player model also gets where the ball was last seen and where it turns up next (a restart is
    taken from a line or a spot; play that went on brings the ball back anywhere, still moving),
    ("ball"), the share of players sprinting and the team's spread ("team") (players keep their shape in play and
    regroup for a set piece)."""
    length, width = pitch
    n = len(t["bx"])
    seen = np.isfinite(t["bx"])
    sx, sy = smooth_nan(t["bx"], 5), smooth_nan(t["by"], 5)
    speed = np.hypot(np.gradient(sx), np.gradient(sy)) * FPS
    speed = np.where(seen, speed, np.nan)
    fast = np.nan_to_num(speed) > 8.0
    outside = np.maximum.reduce([-t["bx"], t["bx"] - length, -t["by"], t["by"] - width,
                                 np.zeros(n)])
    outside = np.where(seen, outside, np.nan)
    to_line = np.minimum.reduce([np.abs(t["bx"]), np.abs(length - t["bx"]),
                                 np.abs(t["by"]), np.abs(width - t["by"])])
    still = np.where(np.isfinite(speed), (speed < 0.5).astype(float), np.nan)
    pspeed = np.nanmean(t["ps"], axis=1)
    d = np.hypot(t["px"] - t["bx"][:, None], t["py"] - t["by"][:, None])
    nearest = np.nanmin(np.where(np.isfinite(d), d, np.nan), axis=1)

    ball, players = {}, {}
    for half in (12, 50, 125):
        ball[f"bspeed{half}"] = _window_mean(speed, half)
        ball[f"still{half}"] = _window_mean(still, half)
    ball["bspeed_max25"] = _window_max(np.nan_to_num(speed), 25)
    ball["outside12"] = _window_max(np.nan_to_num(outside), 12)
    ball["outside75"] = _window_max(np.nan_to_num(outside), 75)
    ball["to_line"] = _window_mean(to_line, 25)
    ball["nearest"] = _window_mean(nearest, 25)
    for half in (12, 50, 125, 250):
        players[f"pspeed{half}"] = _window_mean(pspeed, half)
    players["pspeed_trend_back"] = _window_mean(pspeed, 50) - _window_mean(np.roll(pspeed, 75), 50)
    players["pspeed_trend_fwd"] = _window_mean(np.roll(pspeed, -75), 50) - _window_mean(pspeed, 50)
    if SHORT_TRENDS:
        # the sharp slow-down right after a whistle: the next 2 s against the previous 2 s
        players["pspeed_drop_short"] = (_window_mean(np.roll(pspeed, -25), 25)
                                        - _window_mean(np.roll(pspeed, 25), 25))
        players["pspeed_drop_back"] = (_window_mean(pspeed, 12)
                                       - _window_mean(np.roll(pspeed, 75), 25))
    for half in (12, 50, 125):
        players[f"seen{half}"] = _window_mean(seen.astype(float), half)
    if gap in (True, "all", "ball"):
        line_now = np.where(seen, to_line, np.nan)
        players["gap_prev_line"] = _last_seen(line_now, seen)
        players["gap_next_line"] = _next_seen(line_now, seen)
        players["gap_next_speed"] = _next_seen(np.where(seen, _window_mean(speed, 12), np.nan), seen)
    if gap in (True, "all", "team"):
        players["sprint50"] = _window_mean(np.nanmean(np.where(np.isfinite(t["ps"]), t["ps"] > 4.0, np.nan),
                                                      axis=1), 50)
        spread = np.hypot(np.nanstd(t["px"], axis=1), np.nanstd(t["py"], axis=1))
        players["spread50"] = _window_mean(spread, 50)
    ball_ok = _window_mean(seen.astype(float), 12) >= 0.5
    return ball, players, ball_ok


def features(t, pitch, gap=False):
    """(ball+player features, player-only features, frames where the ball features apply).

    Whether the ball is seen is a feature of the player model only. In both datasets the ball is
    mostly missing during stoppages: in the clean sample data it is removed, and in our automatic
    tracking, frames without a ball have players moving exactly as in stoppages (median 1.05 m/s,
    against 1.21 in known stoppages and 2.33 in known live play). The ball model is only used where
    the ball is seen. The number of tracked players differs between the datasets and is left out.
    """
    ball, players, ball_ok = _feature_table(t, pitch, gap)
    P = np.stack(list(players.values()), axis=1).astype(np.float32)
    BP = np.concatenate([np.stack(list(ball.values()), axis=1).astype(np.float32), P], axis=1)
    return BP, P, ball_ok


def _clf():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=4, max_iter=200, learning_rate=0.05,
                                          min_samples_leaf=200, l2_regularization=1.0,
                                          random_state=0)


def train(games, seed=0, step=5, gap=False):
    """Two classifiers for P(dead), trained on the given validation matches (every `step`-th
    frame): one on ball and player signals for frames where the ball is seen, one on player
    signals alone for everything."""
    from possession.external import PITCH, load_game
    XB, yB, XP, yP = [], [], [], []
    for name in games:
        t, labels, _ = load_game(name, seed=seed)
        BP, P, ok = features(t, PITCH, gap)
        g = np.flatnonzero(labels != "X")[::step]
        dead = labels[g] == "D"
        XP.append(P[g]); yP.append(dead)
        gb = ok[g]
        XB.append(BP[g][gb]); yB.append(dead[gb])
    with_ball, players_only = _clf(), _clf()
    with_ball.fit(np.concatenate(XB), np.concatenate(yB))
    players_only.fit(np.concatenate(XP), np.concatenate(yP))
    return with_ball, players_only, gap


def spread_setting(rng):
    """One random degradation: wide ranges with round bounds, none set from our match. Every degradation
    value that was measured on our match is drawn here, so none reaches the classifier."""
    u = rng.uniform
    chunk = int(u(100, 1000))
    return dict(own_gap_share=u(0, 0.5), ball_smooth=u(0.5, 15), ball_drift=u(0, 3), ball_noise=u(0, 1),
                drift_tau=u(10, 250), drift_smooth=u(0, 25), player_noise=u(0, 1), player_smooth=u(0, 25),
                drop_rate=u(0, 0.2), drop_chunk=(chunk // 10, chunk), waiting_share=u(0, 1),
                waiting_kw=dict(offset=(1.0, u(2, 20)), settle=u(25, 200), lead=(0.0, u(0, 1))),
                after_whistle_share=u(0, 1), after_kw=dict(seconds=(1.0, u(3, 15)), friction=u(0.5, 3)),
                fill_frames=int(rng.integers(0, 26)))


def train_spread(games, seed=0, step=5, gap=False, copies=SPREAD_COPIES):
    """The two classifiers of train(), on the same number of frames, with each training frame taken from
    one of `copies` degraded versions of its match, picked at random. Each version has its own random
    setting (spread_setting). Our match is never read."""
    from possession.external import PITCH, load_spread
    XB, yB, XP, yP = [], [], [], []
    for i, name in enumerate(games):
        pick = None
        for c in range(copies):
            setting = spread_setting(np.random.default_rng([seed, i, c, 8]))
            t, labels = load_spread(name, seed * 1000 + i * 100 + c, setting)
            if pick is None:
                g = np.flatnonzero(labels != "X")[::step]
                pick = np.random.default_rng([seed, i, 7]).integers(0, copies, len(g))
            BP, P, ok = features(t, PITCH, gap)
            rows = g[pick == c]
            dead = labels[rows] == "D"
            XP.append(P[rows]); yP.append(dead)
            gb = ok[rows]
            XB.append(BP[rows][gb]); yB.append(dead[gb])
    with_ball, players_only = _clf(), _clf()
    with_ball.fit(np.concatenate(XB), np.concatenate(yB))
    players_only.fit(np.concatenate(XP), np.concatenate(yP))
    return with_ball, players_only, gap


def p_dead(models, t, pitch):
    with_ball, players_only, gap = models
    BP, P, ok = features(t, pitch, gap)
    out = players_only.predict_proba(P)[:, 1]
    if ok.any():
        out[ok] = with_ball.predict_proba(BP[ok])[:, 1]
    return out
