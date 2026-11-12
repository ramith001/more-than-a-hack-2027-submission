"""Possession model v1.

Three steps:
  1. clean the ball track (fill short gaps)
  2. per-frame log-likelihoods for H, A, D from tracking geometry: who is near the ball while it
     is under control (not in flight), and which team touched it last
  3. Viterbi over the three states. Possession can only change hands cheaply at the moment the
     team taking over touches the ball, so the output is whole possessions with handovers at
     touches rather than when a defender merely gets close
"""

import numpy as np

H, A, D = 0, 1, 2
LABELS = np.array(["H", "A", "D"])

PARAMS = {
    "max_gap": 25,            # ball gaps up to this many frames are interpolated
    "tau": 1.5,               # metres, softness of the nearest-player comparison
    "far": 4.0,               # metres, beyond this nobody is close to the ball
    "flight_speed": 11.0,     # m/s, above this the ball is in flight and proximity is ignored
    "touch_jerk": 1.5,        # m/s, change in ball velocity that counts as a touch
    "touch_reach": 2.5,       # metres, a player this close at that moment made the touch
    "touch_lag": 3,           # frames before a detected touch that also count for who touched it
    "touch_window": 3,        # frames either side of a touch where a handover is cheap
    "touch_hold": 250,        # frames a touch keeps counting as evidence for its team
    "touch_weight": 0.25,      # per-frame evidence for the team that touched it last
    "touch_direction": True, 
    "touch_angle": 10,        # degrees, a touch must turn the ball at least this much (published: 10)
    "touch_dspeed": 5,        # m/s, or change its speed at least this much (published: 5)
    "contested_keeps": False, # True: a touch with both teams within reach cannot hand possession over
    "confirm": "",            # "", "contested" or "all": a touch only hands possession over if the
                              # same team also makes the next touch (they kept it), or nobody
                              # touches it again within confirm_window frames
    "confirm_window": 75,
    "near_weight": 1.0,       # how much being closest to the ball counts, compared with touches
    "scuffle_radius": 2.0,    # metres: while both teams are this close to the ball it is a scuffle,
                              # and being closest says nothing about possession (0 = off)
    "out_margin": 0.5,        # metres outside the lines counts as out of play
    "slow_speed": 1.6,        # m/s, mean player speed below this looks like a stoppage
    "speed_window": 50,       # frames, smoothing for collective speed
    "missing_long": 50,       # frames without the ball before that counts as evidence of dead ball
    "switch_team": 60.0,      # cost of H <-> A away from a touch
    "switch_touch": 10.0,     # cost of H <-> A at a touch by the team taking over
    "switch_dead": 10.0,      # cost of entering or leaving D
    "min_dead": 75,           # frames, shortest possible stoppage (97% of real ones last 3 s+)
    "learned_dead": True,
    "team_evidence": True,    # learned team evidence (possession/team_evidence.py), run.py trains it on external matches
    "use_restarts": True,     # the team taking a tagged set piece has the ball at the restart
    "restart_frames": 12,     # frames after the restart that are live and belong to that team
    "restart_weight": 2.0,    # per-frame evidence for that team
    "throw_ins": True,        # throw-in team from the player holding the ball at the touchline
    "fast_unseen": 0,         # player speed (m/s) above which a lost ball is less likely dead (0 = off)
    "fast_unseen_scale": 0.5,
    "restart_moving": 6.0,    # ball speed (m/s) that marks a set piece being taken
    "restart_alive": 0,       # frames after a detected restart where the ball cannot be dead
    "restart_dead_before": 0, # frames before a detected restart forced dead
    "min_spell": 50,          # frames, shorter possession spells go to the team around them (0 = off)
    "restart_fill": 0,        # seconds: live spells between a stoppage and the tagged restart are dead (0 = off)
    "keep_restart_spells": True,  # a short spell right after a stoppage stays (restarting team)
    "min_live_in_dead": 0,    # frames, shorter live spells between two stoppages become dead (0 = off)
    "blip_max_speed": 5.0,    # m/s: a live spell between two stoppages in which the ball never moves faster
                              # than this is the ball being carried to the spot, not play (0 = off)
    "blip_max_len": 250,      # frames, only spells shorter than this
    "kickoff_seen_only": False,  # force the ball live only after kick-offs seen leaving the spot (io.forced_alive)
    "dead_gap_features": False,  # dead ball classifier also sees where the ball was last seen and turns up
                              # next ("ball"), sprinting and team spread ("team"), or both ("all")
    "goal_kick_exit": 0,      # m/s: inside a stoppage, the ball leaving a goal area faster than this ends it
                              # (a goal kick), the team defending that goal has it (0 = off)
    "play_on_speed": 0,       # m/s: a short stoppage in which the players never slow below this (1 s mean)
                              # and after which the ball turns up away from every line is play going on (0 = off)
    "play_on_line": 8.0,      # metres from the nearest line the ball must turn up at
    "play_on_len": 375,       # frames, only stoppages shorter than this
    "spell_keep_touches": 0,  # a short spell still counts with this many touches by its team (0 = off)
    "follow_weight": 0,       # evidence for the team whose nearest player moves with the ball (0 = off)
    "follow_reach": 2.0,
    "follow_speed": 8.0,
    "reception_path": False,  # at a reception, credit the first player on the ball's path
    "switch_control": 0,      # cost of H <-> A at the start of a spell of clear control (0 = off)
    "control_frames": 25,     # frames of clear control needed
    "control_speed": 6.0,     # m/s, the ball must be slower than this
    "control_reach": 2.5,     # metres, the controlling team's nearest player
    "control_margin": 1.5,    # metres, how much further the other team must be     # use the dead ball classifier (deadball.py) instead of the old rule
}


def rolling_mean(x, w):
    x = np.where(np.isfinite(x), x, 0.0)
    c = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - w)
    hi = np.minimum(len(x), i + w + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def fill_ball(bx, by, max_gap):
    bx, by = bx.astype(float).copy(), by.astype(float).copy()
    ok = np.isfinite(bx)
    idx = np.flatnonzero(ok)
    gaps = np.flatnonzero(np.diff(idx) > 1)
    for g in gaps:
        a, b = idx[g], idx[g + 1]
        if b - a - 1 <= max_gap:
            t = np.linspace(0, 1, b - a + 1)[1:-1]
            bx[a + 1:b] = bx[a] + t * (bx[b] - bx[a])
            by[a + 1:b] = by[a] + t * (by[b] - by[a])
    return bx, by


def run_length_missing(ok):
    """For each frame, how many consecutive frames the ball has been missing (0 if present)."""
    out = np.zeros(len(ok), dtype=np.int32)
    run = 0
    for i, v in enumerate(ok):
        run = 0 if v else run + 1
        out[i] = run
    return out


def smooth_nan(x, k):
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    m = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - k // 2), np.minimum(len(x), i + k // 2 + 1)
    full = (m[hi] - m[lo]) == (hi - lo)
    return np.where(full, (c[hi] - c[lo]) / np.maximum(hi - lo, 1), np.nan)


def ball_kinematics(bx, by, k=4, vectors=False):
    """Ball speed (m/s) and how sharply its velocity changes around each frame (a touch).

    With vectors=True also returns the change in velocity (dvx, dvy): the push the ball got.
    """
    sx, sy = smooth_nan(bx, 5), smooth_nan(by, 5)
    vx, vy = np.gradient(sx) * 25, np.gradient(sy) * 25
    speed = np.hypot(vx, vy)
    dvx, dvy = np.roll(vx, -k) - np.roll(vx, k), np.roll(vy, -k) - np.roll(vy, k)
    jerk = np.hypot(dvx, dvy)
    if vectors:
        return np.nan_to_num(speed), np.nan_to_num(jerk), np.nan_to_num(dvx), np.nan_to_num(dvy)
    return np.nan_to_num(speed), np.nan_to_num(jerk)


def team_distances(t, bx, by):
    """Distance from the ball to every slot, the per-frame team of every slot, and the nearest
    home and away distances. Uses t["team_frame"] (goalkeepers corrected) when present."""
    d = np.hypot(t["px"] - bx[:, None], t["py"] - by[:, None])
    d = np.where(np.isfinite(d), d, np.inf)
    tf = t.get("team_frame")
    if tf is None:
        tf = np.repeat(t["team"][None, :], len(bx), axis=0)
    dh = np.where(tf == 0, d, np.inf).min(1)
    da = np.where(tf == 1, d, np.inf).min(1)
    return d, tf, dh, da


def detect_touches(t, bx, by, p):
    """Frames where a player touches the ball, and the team that touched it.

    A touch is a peak in the change of the ball's velocity with a player within `touch_reach`.
    With `touch_angle` / `touch_dspeed` set, the change must also be a real change of flight: the
    direction turns by more than `touch_angle` degrees or the speed changes by more than
    `touch_dspeed` m/s (the published definition uses 10 degrees and 5 m/s). That stops a fast
    ball's small natural curve counting as a touch by whoever it passes.
    With `touch_direction`, the touch is credited to the player the ball was pushed away from.
    Returns (frames, teams, contested), contested being True where players of both teams were
    within reach.
    """
    d, tf, dh, da = team_distances(t, bx, by)
    sx, sy = smooth_nan(bx, 5), smooth_nan(by, 5)
    vx, vy = np.nan_to_num(np.gradient(sx) * 25), np.nan_to_num(np.gradient(sy) * 25)
    k = 4
    ax_, ay_ = np.roll(vx, k), np.roll(vy, k)          # velocity before
    bx_, by_ = np.roll(vx, -k), np.roll(vy, -k)        # velocity after
    dvx, dvy = bx_ - ax_, by_ - ay_
    jerk = np.hypot(dvx, dvy)
    is_peak = (jerk >= p["touch_jerk"]) & (jerk >= np.roll(jerk, 1)) & (jerk >= np.roll(jerk, -1))
    if p.get("touch_angle") or p.get("touch_dspeed"):
        s_before, s_after = np.hypot(ax_, ay_), np.hypot(bx_, by_)
        cosang = (ax_ * bx_ + ay_ * by_) / np.maximum(s_before * s_after, 1e-6)
        angle = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
        angle = np.where(np.minimum(s_before, s_after) < 1.0, 180.0, angle)
        real = (angle >= p.get("touch_angle", 1e9)) | (np.abs(s_after - s_before) >= p.get("touch_dspeed", 1e9))
        is_peak &= real
    lag = int(p.get("touch_lag", 0))
    if lag:
        # The smoothed ball shows a touch a few frames late, when it has already left the foot.
        # Measure each team's distance at the closest moment in the `lag` frames before the peak.
        from scipy.ndimage import minimum_filter1d
        dh = minimum_filter1d(dh, lag + 1, origin=lag // 2)
        da = minimum_filter1d(da, lag + 1, origin=lag // 2)
        d = minimum_filter1d(d, lag + 1, axis=0, origin=lag // 2)
    frames = np.flatnonzero(is_peak & (np.minimum(dh, da) <= p["touch_reach"]))
    teams = np.where(dh[frames] <= da[frames], H, A).astype(np.int8)
    contested = (dh[frames] <= p["touch_reach"]) & (da[frames] <= p["touch_reach"])
    if p.get("touch_direction"):
        for i, f in enumerate(frames):
            cand = np.flatnonzero(d[f] <= p["touch_reach"])
            if len(cand) < 2 or len(set(tf[f, cand])) < 2:
                continue
            sb, sa = np.hypot(ax_[f], ay_[f]), np.hypot(bx_[f], by_[f])
            if p.get("reception_path") and sa < 0.7 * sb and sb > 3.0:
                # a reception: credit the player the incoming ball reaches first along its path
                ux, uy = ax_[f] / sb, ay_[f] / sb
                along = (t["px"][f, cand] - bx[f]) * ux + (t["py"][f, cand] - by[f]) * uy
                teams[i] = tf[f, cand[np.argmin(along)]]
                continue
            ux, uy = bx[f] - t["px"][f, cand], by[f] - t["py"][f, cand]
            norm = np.hypot(ux, uy) * max(jerk[f], 1e-6) + 1e-6
            align = (ux * dvx[f] + uy * dvy[f]) / norm
            teams[i] = tf[f, cand[np.argmax(align)]]
    return frames, teams, contested


def touch_owner(frames, teams, n, p):
    """Team of the last player to touch the ball, carried forward until the next touch.

    A touch is a sharp change in the ball's velocity with a player within reach. The owner only
    changes at touches, so a ball in flight stays with the team that played it. -1 before the
    first touch and across long stretches with no ball.
    """
    owner = np.full(n, -1, dtype=np.int8)
    nxt = np.r_[frames[1:], n]
    for f, tm, g in zip(frames, teams, nxt):
        owner[f:min(g, f + p["touch_hold"])] = tm
    return owner


def follow_evidence(t, bx, by, p):
    """Evidence from who the ball moves with: a dribbler's ball travels with him, while a player
    pressing him moves differently. For each team's nearest player within `follow_reach`, compare
    his velocity with the ball's over about half a second; the team whose player moves with the
    ball gets up to `follow_weight` of evidence. Only at dribbling speeds."""
    d, tf, dh, da = team_distances(t, bx, by)
    n = len(bx)
    sx, sy = smooth_nan(bx, 5), smooth_nan(by, 5)
    vbx, vby = np.gradient(sx) * 25, np.gradient(sy) * 25
    ppx, ppy = t["px"], t["py"]
    vpx = np.gradient(np.where(np.isfinite(ppx), ppx, np.nan), axis=0) * 25
    vpy = np.gradient(np.where(np.isfinite(ppy), ppy, np.nan), axis=0) * 25
    rows = np.arange(n)
    mis = []
    for team in (H, A):
        dd = np.where(tf == team, d, np.inf)
        dd = np.where(np.isfinite(dd), dd, np.inf)
        j = np.argmin(dd, axis=1)
        close = dd[rows, j] <= p.get("follow_reach", 2.0)
        m = np.hypot(vpx[rows, j] - vbx, vpy[rows, j] - vby)
        m = np.where(close & np.isfinite(m), m, np.nan)
        mis.append(m)
    mh = rolling_mean_nan(mis[0], 6)
    ma = rolling_mean_nan(mis[1], 6)
    speed = np.hypot(vbx, vby)
    use = np.isfinite(mh) & np.isfinite(ma) & (speed > 1.0) & (speed < p.get("follow_speed", 8.0))
    x = np.where(use, np.tanh((ma - mh) / 1.5), 0.0) * p["follow_weight"]
    ev = np.zeros((n, 3))
    ev[:, H] = x / 2
    ev[:, A] = -x / 2
    return ev


def rolling_mean_nan(x, w):
    ok = np.isfinite(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    m = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    cnt = m[hi] - m[lo]
    return np.where(cnt >= w, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def emissions(t, pitch, p=PARAMS):
    length, width = pitch
    bx, by = fill_ball(t["bx"], t["by"], p["max_gap"])
    ok = np.isfinite(bx)
    _, _, dh, da = team_distances(t, bx, by)

    n = len(bx)
    ll = np.zeros((n, 3))

    near = np.minimum(dh, da)
    conf = np.clip((p["far"] - near) / p["far"], 0, 1)
    speed_b, jerk = ball_kinematics(bx, by)
    if p.get("flight_speed"):
        conf = np.where(speed_b > p["flight_speed"], 0.0, conf)
    if p.get("scuffle_radius"):
        conf = np.where(np.maximum(dh, da) <= p["scuffle_radius"], 0.0, conf)
    diff = np.clip((da - dh) / p["tau"], -6, 6)
    ph = 1 / (1 + np.exp(-diff))
    ph = np.where(ok, 0.5 + (ph - 0.5) * conf, 0.5)
    nw = p.get("near_weight", 1.0)
    ll[:, H] = nw * np.log(np.clip(ph, 1e-3, 1))
    ll[:, A] = nw * np.log(np.clip(1 - ph, 1e-3, 1))

    if p.get("touch_weight"):
        frames, teams, _ = detect_touches(t, bx, by, p)
        owner = touch_owner(frames, teams, n, p)
        ll[owner == H, H] += p["touch_weight"]
        ll[owner == A, A] += p["touch_weight"]

    if p.get("follow_weight"):
        ll += follow_evidence(t, bx, by, p)

    out = ok & ((bx < -p["out_margin"]) | (bx > length + p["out_margin"]) |
                (by < -p["out_margin"]) | (by > width + p["out_margin"]))
    speed = rolling_mean(np.nanmean(t["ps"], axis=1), p["speed_window"])
    missing = run_length_missing(np.isfinite(t["bx"]))

    if t.get("p_dead") is not None:
        pd_ = t["p_dead"]
    else:
        dead = np.full(n, -1.0)
        dead += 3.0 * out
        dead += np.clip((p["slow_speed"] - speed) * 2.0, -1.5, 2.5)
        dead += np.clip((missing - p["missing_long"]) / 50.0, 0, 2.0)
        pd_ = 1 / (1 + np.exp(-dead))
    if p.get("fast_unseen"):
        # ball lost while the players are still running: more likely live play than a stoppage
        fast = (~ok) & (rolling_mean(np.nanmean(t["ps"], axis=1), 50) > p["fast_unseen"])
        pd_ = np.where(fast, pd_ * p.get("fast_unseen_scale", 0.5), pd_)
    ll[:, D] = np.log(np.clip(pd_, 1e-3, 1))
    alive = np.log(np.clip(1 - pd_, 1e-3, 1))
    ll[:, H] += alive
    ll[:, A] += alive
    if p.get("team_weight") and t.get("p_team") is not None:
        # learned team evidence (possession/team_evidence.py): P(home has the ball), 0.5 where the ball is unseen
        ph = np.where(t["p_ok"], t["p_team"], 0.5)
        ll[:, H] += p["team_weight"] * np.log(np.clip(ph, 1e-3, 1))
        ll[:, A] += p["team_weight"] * np.log(np.clip(1 - ph, 1e-3, 1))
    return ll


def viterbi(ll, p=PARAMS, into=None):
    """Best H/A/D path.

    `into` (n, 2) optionally gives a per-frame cost of switching into H or A, so a handover can be
    cheap at the moment of a touch and expensive everywhere else.

    Dead ball is a chain of `min_dead` states that must be walked through in order, so every
    stoppage lasts at least that many frames. Without it the path could hand possession over
    through a one-frame "stoppage" whenever that is cheaper than a direct switch.
    """
    n = len(ll)
    if into is None:
        into = np.full((n, 2), float(p["switch_team"]))
    k = max(1, int(p.get("min_dead", 1)))
    sd = p["switch_dead"]
    ns = 2 + k                      # 0 = H, 1 = A, 2..2+k-1 = dead ball, last one absorbing
    emit = np.concatenate([ll[:, :2], np.repeat(ll[:, 2:3], k, axis=1)], axis=1)
    score = np.full(ns, -np.inf)
    score[:3] = emit[0, :3]
    back = np.zeros((n, ns), dtype=np.int16)
    last = ns - 1
    for i in range(1, n):
        new = np.empty(ns)
        bk = np.empty(ns, dtype=np.int16)
        for team, other in ((0, 1), (1, 0)):
            c = (score[team], score[other] - into[i, team], score[last] - sd)
            j = int(np.argmax(c))
            new[team], bk[team] = c[j], (team, other, last)[j]
        c = (score[0] - sd, score[1] - sd)
        j = int(np.argmax(c))
        new[2], bk[2] = c[j], j
        if k > 1:
            new[3:ns] = score[2:ns - 1]
            bk[3:ns] = np.arange(2, ns - 1)
            if score[last] > new[last]:
                new[last], bk[last] = score[last], last
        else:
            if score[2] > new[2]:
                new[2], bk[2] = score[2], 2
        score = new + emit[i]
        back[i] = bk
    path = np.zeros(n, dtype=np.int16)
    path[-1] = int(np.argmax(score))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return np.minimum(path, 2).astype(np.int8)


def switch_costs(t, p=PARAMS):
    """Per-frame cost of switching possession into H or into A.

    Possession changes hands when the other team touches the ball, so a switch into a team is
    cheap within a few frames of a touch by that team and costs the full `switch_team` otherwise.
    """
    n = len(t["bx"])
    into = np.full((n, 2), float(p["switch_team"]))
    if not p.get("switch_touch"):
        return into
    bx, by = fill_ball(t["bx"], t["by"], p["max_gap"])
    w = p.get("touch_window", 3)
    if p.get("switch_control"):
        for f, k in zip(*control_events(t, p)):
            into[max(0, f - w):f + w + 1, k] = np.minimum(into[max(0, f - w):f + w + 1, k],
                                                          p["switch_control"])
    frames, teams, contested = detect_touches(t, bx, by, p)
    nxt_frame = np.r_[frames[1:], np.iinfo(np.int64).max]
    nxt_team = np.r_[teams[1:], -1]
    kept = (nxt_team == teams) | (nxt_frame - frames > p.get("confirm_window", 75))
    for f, k, c, ok in zip(frames, teams, contested, kept):
        if c and p.get("contested_keeps"):
            continue
        mode = p.get("confirm", "")
        if not ok and (mode == "all" or (mode == "contested" and c)):
            continue
        into[max(0, f - w):f + w + 1, k] = p["switch_touch"]
    return into


def control_events(t, p=PARAMS):
    """Frames where a team starts a spell of clear control, and which team.

    Control: for `control_frames` frames in a row the ball is seen and slow (under
    `control_speed` m/s), one team's nearest player is within `control_reach` metres, and the
    other team's nearest player is at least `control_margin` metres further away. This catches a
    team winning the ball when the tracking puts the touch itself too far away to register.
    """
    bx, by = t["bx"], t["by"]
    _, _, dh, da = team_distances(t, bx, by)
    speed, _ = ball_kinematics(bx, by)
    seen = np.isfinite(bx)
    slow = seen & (speed < p["control_speed"])
    k = p["control_frames"]
    out_f, out_t = [], []
    for team, mine, theirs in ((H, dh, da), (A, da, dh)):
        ok = slow & (mine <= p["control_reach"]) & (theirs >= mine + p["control_margin"])
        run = np.convolve(ok.astype(int), np.ones(k, dtype=int), mode="full")[k - 1:len(ok) + k - 1]
        full = run >= k
        starts = np.flatnonzero(full & ~np.r_[False, full[:-1]])
        out_f.append(starts)
        out_t.append(np.full(len(starts), team, dtype=np.int8))
    f = np.concatenate(out_f)
    order = np.argsort(f)
    return f[order], np.concatenate(out_t)[order]


def predict(t, pitch, blocks, n, p=PARAMS):
    ll = emissions(t, pitch, p)
    for a, b in t.get("forced_dead", []):
        ll[a:b + 1, H] = ll[a:b + 1, A] = -50.0
        ll[a:b + 1, D] = 0.0
    for a, b in t.get("forced_alive", []):
        ll[a:b + 1, D] = -50.0
    if p.get("use_restarts") and t.get("restarts"):
        from possession import restarts
        ll = restarts.apply(ll, t["restarts"], p)
    into = switch_costs(t, p)
    labels = np.full(n, "X", dtype="<U1")
    keep = None
    if p.get("min_spell") and p.get("spell_keep_touches"):
        bx, by = fill_ball(t["bx"], t["by"], p["max_gap"])
        keep = detect_touches(t, bx, by, p)[:2]
    for _, a, b in blocks:
        path = viterbi(ll[a:b + 1], p, into[a:b + 1])
        if p.get("min_spell"):
            tch = None
            if keep is not None:
                inb = (keep[0] >= a) & (keep[0] <= b)
                tch = (keep[0][inb] - a, keep[1][inb])
            path = drop_short_spells(path, int(p["min_spell"]), tch,
                                     int(p.get("spell_keep_touches", 0)),
                                     int(p.get("min_live_in_dead", 0)),
                                     bool(p.get("keep_restart_spells", True)))
        labels[a:b + 1] = LABELS[path]
    if p.get("goal_kick_exit"):
        labels = goal_kick_exits(t, pitch, labels, float(p["goal_kick_exit"]))
    if p.get("play_on_speed"):
        labels = play_on(t, pitch, labels, float(p["play_on_speed"]), float(p.get("play_on_line", 8.0)),
                         int(p.get("play_on_len", 375)))
    if p.get("blip_max_speed"):
        labels = merge_blips(t, labels, float(p["blip_max_speed"]), int(p.get("blip_max_len", 250)))
    if p.get("restart_fill"):
        labels = fill_until_restart(labels, t.get("restarts") or [], int(p["restart_fill"] * 25))
    return labels


def goal_kick_exits(t, pitch, labels, min_speed, within=25):
    """A goal kick is taken from inside the goal area and is in play once it is kicked and clearly
    moves (Law 16). Inside a predicted stoppage, the first frame where the seen ball is in a goal
    area and within `within` frames is outside it moving faster than `min_speed` ends the stoppage;
    the rest of it goes to the team defending that goal (the team taking the kick), whose players
    are on average nearer to it."""
    length, width = pitch
    speed, _ = ball_kinematics(t["bx"], t["by"])
    bx, by = t["bx"], t["by"]
    in_area = np.isfinite(bx) & (np.abs(by - width / 2) <= 9.16) & ((bx <= 5.5) | (bx >= length - 5.5))
    out_fast = np.isfinite(bx) & ~in_area & (speed > min_speed)
    labels = labels.copy()
    ch = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    en = np.r_[ch[1:], len(labels)]
    for a, b in zip(ch, en):
        if labels[a] != "D":
            continue
        for f in np.flatnonzero(in_area[a:b]) + a:
            if not out_fast[f + 1:min(f + within + 1, len(labels))].any():
                continue
            team_x = [np.nanmean(t["px"][f, t["team"] == k]) for k in (0, 1)]
            near_left = bx[f] < length / 2
            team = int(np.argmin(team_x)) if near_left else int(np.argmax(team_x))
            labels[f:b] = "HA"[team]
            break
    return labels


def play_on(t, pitch, labels, min_speed, line, max_len, look=(25, 50)):
    """A stoppage the tracking makes up when it loses the ball while play goes on. Play stops with a
    whistle or the ball going out, and the players then drop to walking pace; a restart is taken
    from a line (throw-in, corner, goal kick) or a spot. So a dead spell between two live spells,
    shorter than `max_len` frames, in which the players' mean speed (over 1 s) never falls below
    `min_speed`, and after which the ball turns up more than `line` metres from every line, is
    live. With the ball unseen nobody knows when it changed hands: the first half goes to the team
    before, the second half to the team after."""
    length, width = pitch
    ps = rolling_mean_nan(np.nanmean(t["ps"], axis=1), 12)
    seen = np.isfinite(t["bx"])
    labels = labels.copy()
    ch = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    en = np.r_[ch[1:], len(labels)]
    for i in range(1, len(ch) - 1):
        a, b = ch[i], en[i]
        if labels[a] != "D" or labels[a - 1] not in "HA" or labels[b] not in "HA" or b - a >= max_len:
            continue
        w = np.arange(max(a, b - look[0]), min(len(seen), b + look[1]))
        w = w[seen[w]]
        if not len(w) or not np.nanmin(ps[a:b]) >= min_speed:
            continue
        f = w[0]
        if min(t["bx"][f], length - t["bx"][f], t["by"][f], width - t["by"][f]) <= line:
            continue
        mid = (a + b) // 2
        labels[a:mid], labels[mid:b] = labels[a - 1], labels[b]
    return labels


def merge_blips(t, labels, max_speed, max_len):
    """Play is dead from the whistle until a player restarts it, and nobody gains possession while
    the ball is carried to the spot. A live spell between two stoppages, shorter than `max_len`
    frames, in which the ball never moves faster than `max_speed` (no kick, no pass, so no restart)
    is part of the stoppage."""
    speed, _ = ball_kinematics(t["bx"], t["by"])
    labels = labels.copy()
    ch = np.flatnonzero(np.r_[True, labels[1:] != labels[:-1]])
    en = np.r_[ch[1:], len(labels)]
    for i in range(1, len(ch) - 1):
        a, b = ch[i], en[i]
        if (labels[a] in "HA" and labels[a - 1] == "D" and labels[b] == "D" and b - a < max_len
                and np.max(speed[a:b]) < max_speed):
            labels[a:b] = "D"
    return labels


def fill_until_restart(labels, restarts, max_wait):
    """Play stays stopped until the restart (CHALLENGE.md: possession is only live once play has
    been let go on, and it restarts with a clear possessor).

    restarts: [(r, team)] exact restart frames (a set piece seen being taken). Live labels between
    the start of the stoppage just before r (at most `max_wait` frames back) and r are dead: players
    carrying the ball to the spot are not playing.
    """
    labels = labels.copy()
    for r, team in restarts:
        lo = max(0, r - max_wait)
        dead = np.flatnonzero(labels[lo:r] == "D")
        if not len(dead):
            continue
        e = lo + dead[-1] + 1            # end of the last stoppage before the restart
        if e < r:
            labels[e:r] = np.where(np.isin(labels[e:r], ["H", "A"]), "D", labels[e:r])
    return labels


def drop_short_spells(path, m, touches=None, keep_touches=0, live_in_dead=0, keep_restart=True):
    """A possession spell shorter than m frames is not a real change of possession (a touch with no
    control): give it to the team that had the ball before it, or after it when before is dead.
    With `touches` (frames, teams) and `keep_touches` k, a short spell is kept when its team touches
    the ball at least k times in it (Opta: a won ball only starts a possession if a pass or dribble
    follows). With `live_in_dead` n, a spell of live play shorter than n frames between two
    stoppages is part of one stoppage (players walking about with the ball during a long wait).
    With `keep_restart`, a spell straight after a stoppage is kept however short: CHALLENGE.md says
    play always restarts with a clear possessor, so the team taking the restart has the ball until
    the other team gains control."""
    path = path.copy()
    ch = np.flatnonzero(np.diff(path)) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(path)]
    for i, (a, b) in enumerate(zip(st, en)):
        if path[a] == D or i == 0 or i == len(st) - 1:
            continue
        before, after = path[st[i - 1]], path[st[i + 1]]
        if live_in_dead and before == D and after == D and b - a < live_in_dead:
            path[a:b] = D
            continue
        if b - a >= m or (keep_restart and before == D):
            continue
        if touches is not None and keep_touches:
            fr, tm = touches
            if np.sum((fr >= a) & (fr < b) & (tm == path[a])) >= keep_touches:
                continue
        before, after = path[st[i - 1]], path[st[i + 1]]
        if before != D:
            path[a:b] = before
        elif after != D:
            path[a:b] = after
    return path
