"""Validation matches built from Metrica's public sample data.

https://github.com/metrica-sports/sample-data has three full matches with corrected tracking and
event data. For each one this module:

  1. loads the tracking into the same arrays as possession/io.py (metres, 46-style slot layout)
  2. derives frame-by-frame H/A/D/X labels from the event data
  3. degrades the tracking so it looks like the automatic tracking we are given:
     the ball is removed using real missing-ball stretches copied from our own match,
     ball and player positions get noise, and player tracks drop out in chunks

The labels are a proxy for the competition ground truth, not a copy of it. They are good for
comparing two versions of the model against each other, not for predicting our final score.

    python possession/external.py           # downloads the data if needed, prints a summary
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.io import load_tracking, graded_mask  # noqa: E402

EXT_DIR = os.path.join(ROOT, "data", "external")
REPO = "https://github.com/metrica-sports/sample-data.git"
COMMIT = "e706dd506b360d69d9d123d5b8026e7294b13996"   # pinned so upstream changes cannot alter the bench
PITCH = (105.0, 68.0)
FPS = 25
GAMES = ("Sample_Game_1", "Sample_Game_2", "Sample_Game_3")


def _head(target):
    if not os.path.isdir(os.path.join(target, ".git")):
        return None   # otherwise git would report the enclosing repo's commit
    try:
        out = subprocess.run(["git", "-C", target, "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def ensure_downloaded():
    target = os.path.join(EXT_DIR, "sample-data")
    if not os.path.isdir(os.path.join(target, "data")):
        # fetch into a temporary folder and move it into place, so a failed download can be retried
        tmp = target + ".partial"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["git", "init", "-q", tmp], check=True)
        subprocess.run(["git", "-C", tmp, "remote", "add", "origin", REPO], check=True)
        subprocess.run(["git", "-C", tmp, "fetch", "-q", "--depth", "1", "origin", COMMIT], check=True)
        subprocess.run(["git", "-C", tmp, "checkout", "-q", "FETCH_HEAD"], check=True)
        if os.path.isdir(target) and not os.listdir(target):
            os.rmdir(target)
        os.rename(tmp, target)
    else:
        head = _head(target)
        if head != COMMIT:
            print(f"warning: {target} is at {head or 'an unknown commit (not a git repo?)'}, "
                  f"expected {COMMIT}; using it as is", file=sys.stderr)
    return os.path.join(target, "data")


# ---------------------------------------------------------------- tracking

def _read_csv_team(path):
    with open(path) as f:
        rows = list(csv.reader(f))
    header = rows[2]
    body = np.array([[np.nan if v in ("", "NaN") else float(v) for v in r] for r in rows[3:]])
    player_cols = [i for i, h in enumerate(header) if h.startswith("Player")]
    ball_col = header.index("Ball")
    period = body[:, 0].astype(int)
    xs = body[:, player_cols]
    ys = body[:, [c + 1 for c in player_cols]]
    return period, xs, ys, body[:, ball_col], body[:, ball_col + 1]


def _load_csv_game(folder, name):
    period, hx, hy, bx, by = _read_csv_team(os.path.join(folder, name + "_RawTrackingData_Home_Team.csv"))
    _, ax, ay, _, _ = _read_csv_team(os.path.join(folder, name + "_RawTrackingData_Away_Team.csv"))
    px = np.concatenate([hx, ax], axis=1)
    py = np.concatenate([hy, ay], axis=1)
    team = np.array([0] * hx.shape[1] + [1] * ax.shape[1], dtype=np.int8)
    return period, px, py, bx, by, team


def _load_epts_game(folder, name):
    root = ET.parse(os.path.join(folder, name + "_metadata.xml")).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    team_of_player = {p.get("id"): 0 if p.get("teamId") == team_ids[0] else 1
                      for p in root.iter("Player")}
    player_of_channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    layouts = set()
    for spec in root.iter("DataFormatSpecification"):
        refs = [r.get("playerChannelId") for r in spec.iter("PlayerChannelRef")]
        layouts.add(tuple(team_of_player[player_of_channel[r]] for r in refs if r.endswith("_x")))
    if len(layouts) != 1:
        raise ValueError("team layout of tracking columns changes during the match")
    team = np.array(layouts.pop(), dtype=np.int8)
    rows = []
    with open(os.path.join(folder, name + "_tracking.txt")) as f:
        for line in f:
            frame, players, ball = line.rstrip("\n").split(":")
            vals = players.replace(";", ",").split(",") + ball.split(",")[:2]
            rows.append([float(v) for v in vals])
    data = np.array(rows)
    px = data[:, 0:-2:2]
    py = data[:, 1:-2:2]
    return None, px, py, data[:, -2], data[:, -1], team


def _speed(px, py):
    vx = np.gradient(px, axis=0) * FPS
    vy = np.gradient(py, axis=0) * FPS
    s = np.hypot(vx, vy)
    k = 5
    c = np.cumsum(np.nan_to_num(s), axis=0)
    out = np.full_like(s, np.nan)
    out[k:] = (c[k:] - c[:-k]) / k
    return np.where(np.isfinite(px), out, np.nan)


# ---------------------------------------------------------------- events -> labels

def _events(folder, name):
    out = []
    if name == "Sample_Game_3":
        d = json.load(open(os.path.join(folder, name + "_events.json")))
        meta = ET.parse(os.path.join(folder, name + "_metadata.xml")).getroot()
        home_id = [t.get("id") for t in meta.iter("Team")][0]
        for e in d["data"]:
            out.append({
                "team": 0 if e["team"]["id"] == home_id else 1,
                "type": e["type"]["name"],
                "subtype": (e.get("subtypes") or {}).get("name", "") if isinstance(e.get("subtypes"), dict)
                else " ".join(s["name"] for s in (e.get("subtypes") or [])),
                "period": e["period"],
                "start": e["start"]["frame"] - 1,
                "end": e["end"]["frame"] - 1,
            })
    else:
        for e in csv.DictReader(open(os.path.join(folder, name + "_RawEventsData.csv"))):
            out.append({
                "team": 0 if e["Team"] == "Home" else 1,
                "type": e["Type"],
                "subtype": e["Subtype"],
                "period": int(e["Period"]),
                "start": int(e["Start Frame"]) - 1,
                "end": int(e["End Frame"]) - 1,
            })
    return out


CONTROL = ("PASS", "RECOVERY", "SHOT", "SET PIECE", "BALL LOST")


def is_restart(e):
    """A set piece that restarts play. Metrica logs goal kicks as the pass (or ball lost, or ball
    out) they turn into, with GOAL KICK in the subtype, never as a SET PIECE."""
    return e["type"] == "SET PIECE" or "GOAL KICK" in e["subtype"]


def void_uncontrolled(events):
    """Mark recoveries that never became possession, following the team's possession rules.

    - a clearance that goes to nobody (a recovery followed by the same team's ball lost or ball out
      tagged CLEARANCE) is not possession; a clearance that reaches a teammate is a pass and counts
    - a save is possession only if the keeper holds it: a SAVED recovery followed straight away by
      the same team's ball lost or ball out (a parry) is not possession
    Challenges in between are skipped. Returns the set of event indices to ignore for ownership.
    """
    void = set()
    order = [i for i in range(len(events)) if events[i]["type"] != "CHALLENGE"]
    for a, b in zip(order, order[1:]):
        e, nx = events[a], events[b]
        if e["type"] != "RECOVERY" or nx["team"] != e["team"] or nx["type"] not in ("BALL LOST", "BALL OUT"):
            continue
        if "CLEARANCE" in nx["subtype"] or "SAVED" in e["subtype"]:
            void.update((a, b))
    return void


TACKLE_GIVES_POSSESSION = False   # old rule, removed: a won tackle no longer hands possession over


def labels_from_events(events, n):
    """H/A/D/X per frame.

    Possession belongs to the team of the latest controlling event (a pass, recovery, shot, set
    piece, or the start of a lost ball, which shows that team had it). It changes hands at the
    new team's first controlling event, so a challenge, a tackle that knocks the ball loose or a
    touch without control does not change it: CHALLENGE.md says that while the ball is contested or
    loose, possession stays with the last team that had it, and a change needs controlled
    possession. Clearances to nobody and parried saves are not possession (see void_uncontrolled).
    The ball is dead from the moment it goes out, a foul, or a goal, until the next set piece
    (goal kicks included, see is_restart).
    The event data has no whistle time, so a foul counts from the foul itself. Frames before each
    kick-off and after each period's last event are X.
    """
    lab = np.full(n, "X", dtype="<U1")
    events = sorted(events, key=lambda e: (e["start"], 0 if is_restart(e) else 1))
    void = void_uncontrolled(events)
    events = [dict(e, void=i in void) for i, e in enumerate(events)]
    for period in sorted({e["period"] for e in events}):
        ev = [e for e in events if e["period"] == period]
        kick = next(e for e in ev if e["type"] == "SET PIECE")
        stop = max(e["end"] for e in ev)
        owner, dead_from, cursor = kick["team"], None, kick["start"]
        for e in ev:
            s = max(e["start"], kick["start"])
            if s > cursor:
                lab[cursor:s] = "D" if dead_from is not None else "HA"[owner]
                cursor = s
            if is_restart(e):
                dead_from = None
                owner = e["team"]
            elif dead_from is None and e["type"] in CONTROL and not e["void"]:
                owner = e["team"]
            elif (TACKLE_GIVES_POSSESSION and dead_from is None and e["type"] == "CHALLENGE"
                  and "WON" in e["subtype"] and "AERIAL" not in e["subtype"]
                  and "FAULT" not in e["subtype"]):
                owner = e["team"]
            if dead_from is None:
                if e["type"] == "BALL OUT":
                    end = max(e["end"], s)
                    lab[cursor:end] = "HA"[owner]
                    cursor, dead_from = end, end
                elif e["type"] == "FAULT RECEIVED":
                    dead_from = s
                elif e["type"] == "SHOT" and ("GOAL" in e["subtype"] or "OUT" in e["subtype"]):
                    end = max(e["end"], s)
                    lab[cursor:end] = "HA"[owner]
                    cursor, dead_from = end, end
        lab[cursor:stop + 1] = "D" if dead_from is not None else "HA"[owner]
    return lab


def require_control(lab, events, seconds, fps=FPS):
    """CHALLENGE.md: possession only changes hands when the new team has controlled possession,
    meaning a player keeps the ball for a few seconds or the team completes more than two
    consecutive passes. A spell won straight from the other team that is shorter than `seconds`
    and has fewer than three passes by the new team goes back to the team that had the ball.
    Spells after a stoppage (set pieces) are left alone."""
    lab = lab.copy()
    passes = np.array([e["start"] for e in events if e["type"] == "PASS"])
    pteam = np.array([e["team"] for e in events if e["type"] == "PASS"])
    i, n = 0, len(lab)
    while i < n:
        j = i
        while j < n and lab[j] == lab[i]:
            j += 1
        if lab[i] in "HA" and i > 0 and lab[i - 1] in "HA" and lab[i - 1] != lab[i]:
            team = "HA".index(lab[i])
            np_ = int(np.sum((passes >= i) & (passes < j) & (pteam == team)))
            if j - i < seconds * fps and np_ < 3:
                lab[i:j] = lab[i - 1]
                k = i
                while k > 0 and lab[k - 1] == lab[i]:
                    k -= 1
                i = k
                continue
        i = j
    return lab


CONTROL_SECONDS = 1.5  # seconds of control for a change of possession on the bench (0 = off); see section 5 of docs/technical_document.md
BALL_FILL_FRAMES = 13  # fill the clean ball's gaps shorter than this in live play before degrading (0 = off); the clean data drops the ball under 1 s about 56 times per 90 min, our tracking about 14
LABEL_FIXES = True    # on since 3 Nov (passed the label-change bar; items A and B, see fix_unlogged): dead before unlogged restarts, restart at missed restarts
WAITING_SHARE = 0.25  # share of stoppages that keep a placed, visible ball (see _waiting_ball)
AFTER_WHISTLE_SHARE = 0.0  # share of stoppages where the ball stays tracked after the whistle (see _after_whistle_ball)


def _runs(m):
    """Start and end (exclusive) of each stretch where the boolean array `m` is True."""
    e = np.diff(np.concatenate(([0], m.astype(int), [0])))
    return np.flatnonzero(e == 1), np.flatnonzero(e == -1)


def fix_unlogged(events, n, bx):
    """Events with the two label fixes (items A and B), as a new list.

    A: a restart that follows live play was preceded by a stoppage the event data never logged,
    so a synthetic FAULT RECEIVED goes at the end of the last event before it (kick-offs excepted).
    B: a stoppage that ends with no logged restart gets one: the first pass or recovery at least
    2 s into it, with the clean ball `bx` seen at that frame, becomes a synthetic SET PIECE.
    """
    ev = sorted(events, key=lambda e: e["start"])
    base = labels_from_events(ev, n)
    out = list(ev)
    for k, e in enumerate(ev):
        if is_restart(e) and e["start"] > 0 and base[e["start"] - 1] in "HA" and "KICK OFF" not in e["subtype"]:
            p = [x for x in ev[:k] if x["start"] < e["start"] - 5 and x["type"] != "CARD"]
            if p:
                f = max(p[-1]["start"], p[-1]["end"])
                out.append(dict(team=p[-1]["team"], type="FAULT RECEIVED", subtype="SYNTH", period=p[-1]["period"], start=f, end=f))
    out = [dict(e) for e in out]
    lab = labels_from_events(out, n)
    for s, e in zip(*_runs(lab == "D")):
        c = [x for x in out if s + 2 * FPS <= x["start"] < e and x["type"] in ("PASS", "RECOVERY")
             and not is_restart(x) and np.isfinite(bx[x["start"]])]
        if c:
            c[0]["type"], c[0]["subtype"] = "SET PIECE", "SYNTH"
    return out


def bench_events(name, bx):
    """The events the bench labels and the control rule use: events_for, plus the label fixes when
    LABEL_FIXES is on. `bx` is the clean, unfilled ball."""
    events = events_for(name)
    return fix_unlogged(events, len(bx), bx) if LABEL_FIXES else events


def label_cache(name):
    """Path of the label cache. v4: bench control 1.5 s and ball fill; v5: label fixes A and B.
    (v2: goal kicks end a stoppage; v3: no tackle rule.) The control rule and the fill run after the cache."""
    return os.path.join(EXT_DIR, name + ("_v5.npz" if LABEL_FIXES else "_v4.npz"))


def _fill_short(bx, by, lab, maxlen):
    """Linear fill of the ball's gaps shorter than `maxlen` frames that are not at the array edges
    and have live play (H or A) at both ends."""
    bx, by = bx.copy(), by.copy()
    for a, b in zip(*_runs(~np.isfinite(bx))):
        if b - a >= maxlen or a == 0 or b >= len(bx):
            continue
        if lab[a - 1] in "HA" and lab[b] in "HA":
            w = (np.arange(a, b) - (a - 1)) / (b - (a - 1))
            bx[a:b] = bx[a - 1] + w * (bx[b] - bx[a - 1])
            by[a:b] = by[a - 1] + w * (by[b] - by[a - 1])
    return bx, by


# ---------------------------------------------------------------- degradation

def _smooth_visible(x, sigma):
    """Gaussian smoothing applied separately to each stretch where the ball is visible."""
    from scipy.ndimage import gaussian_filter1d
    out = x.copy()
    edges = np.diff(np.concatenate(([0], np.isfinite(x).astype(int), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if b - a > 3:
            out[a:b] = gaussian_filter1d(x[a:b], sigma, mode="nearest")
    return out


def _drift(n, sigma, tau, rng, smooth=0):
    """Slowly wandering error: standard deviation `sigma` metres, changing over about `tau` frames.
    With `smooth` > 0 the error is also smoothed (Gaussian, frames), so it wanders without adding
    frame-to-frame jitter."""
    a = np.exp(-1.0 / tau)
    e = rng.normal(0, sigma * np.sqrt(1 - a * a), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = a * x[i - 1] + e[i]
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        x = gaussian_filter1d(x, smooth)
        x *= sigma / max(x.std(), 1e-9)
    return x


def _smooth_noise(rng, sigma, shape, smooth):
    """Position error of standard deviation `sigma` that changes smoothly over about `smooth`
    frames (our player tracks have well under a centimetre of frame-to-frame jitter)."""
    e = rng.normal(0, 1, shape)
    if smooth:
        from scipy.ndimage import gaussian_filter1d
        e = gaussian_filter1d(e, smooth, axis=0)
        e /= max(e.std(), 1e-9)
    return sigma * e


def _waiting_ball(t, labels, seed, share, offset=(1.0, 8.0), settle=75, lead=(0.0, 0.3)):
    """Clean ball with a placed, mostly still ball through a share of the stoppages.

    During a set piece wait the ball is brought to the restart spot and sits there until it is
    taken. The clean sample data drops the ball for almost all dead time, but our tracking keeps it
    through about a quarter of set piece waits (seen for over 80% of the wait in 26% of them, at a
    median 1.65 m/s and 3.6 m from the spot). For a random `share` of the stoppages, the missing
    ball is filled in from a random point in the first `lead` of the stoppage up to the restart:
    it starts `offset` metres from the restart spot and settles onto it over about `settle` frames.
    """
    rng = np.random.default_rng([seed, 1])     # its own stream: the other damage stays the same
    bx, by = t["bx"].copy(), t["by"].copy()
    length, width = PITCH
    dead = labels == "D"
    edges = np.diff(np.concatenate(([0], dead.astype(int), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.random() >= share:
            continue
        after = np.flatnonzero(np.isfinite(bx[b:b + 50]))
        if not len(after):
            continue
        x0, y0 = bx[b + after[0]], by[b + after[0]]
        s = a + int(rng.uniform(*lead) * (b - a))
        k = np.arange(b - s)
        r, ang = rng.uniform(*offset), rng.uniform(0, 2 * np.pi)
        fade = np.exp(-k / settle)
        x = np.clip(x0 + r * np.cos(ang) * fade, -2.0, length + 2.0)
        y = np.clip(y0 + r * np.sin(ang) * fade, -2.0, width + 2.0)
        gap = ~np.isfinite(bx[s:b])
        bx[s:b][gap], by[s:b][gap] = x[gap], y[gap]
    return bx, by


def _after_whistle_ball(bx, by, labels, seed, share, seconds=(2.0, 10.0), friction=1.0):
    """The ball stays tracked for a few seconds after a stoppage starts, for a share of stoppages.

    In the clean sample data the ball vanishes the moment play stops. On our match the tracker
    keeps it after a foul or offside whistle while it rolls on and is picked up (in a quarter of
    stoppages it is still seen 3 s after the model's stoppage start, against 1 to 2% in the sample
    data). For a random `share` of the stoppages, the ball carries on from its last live position
    with its last velocity, slowing down (time constant `friction` s), for a random number of seconds.
    """
    rng = np.random.default_rng([seed, 2])
    bx, by = bx.copy(), by.copy()
    dead = labels == "D"
    edges = np.diff(np.concatenate(([0], dead.astype(int), [0])))
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if rng.random() >= share or a < 6:
            continue
        seen = np.flatnonzero(np.isfinite(bx[a - 6:a]))
        if len(seen) < 2:
            continue
        x0, y0 = bx[a - 6 + seen[-1]], by[a - 6 + seen[-1]]
        vx = (bx[a - 6 + seen[-1]] - bx[a - 6 + seen[0]]) / max(seen[-1] - seen[0], 1)
        vy = (by[a - 6 + seen[-1]] - by[a - 6 + seen[0]]) / max(seen[-1] - seen[0], 1)
        n = min(int(rng.uniform(*seconds) * FPS), b - a)
        k = np.arange(1, n + 1)
        tau = friction * FPS
        x = x0 + vx * tau * (1 - np.exp(-k / tau))
        y = y0 + vy * tau * (1 - np.exp(-k / tau))
        gap = ~np.isfinite(bx[a:a + n])
        bx[a:a + n][gap], by[a:a + n][gap] = x[gap], y[gap]
    return bx, by


def degrade(t, seed=0, ball_noise=0.3, ball_smooth=6, ball_drift=0.9, drift_tau=50,
            player_noise=0.15, drop_chunk=(50, 500), drop_rate=0.07, keep_gaps=0.2,
            drift_smooth=8, player_smooth=12, labels=None, waiting_share=0.0,
            after_whistle_share=0.0, fill_frames=0, own_gap_share=None, waiting_kw=None, after_kw=None):
    """Make clean tracking look like the automatic tracking we are given.

    Our ball track is heavily smoothed by the automatic pipeline (about 1 cm of frame-to-frame
    jitter, and touches blurred over several frames). Smoothing the clean ball with a Gaussian of
    `ball_smooth` frames reproduces that: the share of sharp velocity changes and the top ball
    speeds then match our match.

    Our tracked ball is also placed less precisely relative to the players: on clear kicks the
    nearest player is a median 0.89 m from it, and only about a third are within 0.5 m. A slowly
    wandering error of `ball_drift` metres (changing over `drift_tau` frames) reproduces that.

    The clean ball is already missing for most dead-ball time, which matches our match well.
    On top of that, a share `keep_gaps` of the real missing-ball stretches from our match is
    copied in, so live play also loses the ball in realistic gap lengths.

    With `labels` and `waiting_share`, a placed ball is first put back through that share of the
    stoppages (see _waiting_ball).

    With `labels` and `fill_frames`, the clean ball's gaps shorter than that in live play are
    linearly filled first, because the clean data loses the ball for under a second far more often
    than our tracking does (see _fill_short). This uses no random numbers.

    With `own_gap_share`, our match is not read: the gaps are the clean ball's own missing stretches on
    graded frames (after the fill), moved by a random offset, each kept with that probability. It needs
    `labels`. `waiting_kw` and `after_kw` go to _waiting_ball and _after_whistle_ball.
    """
    if labels is not None and fill_frames:
        t = dict(t, **dict(zip(("bx", "by"), _fill_short(t["bx"], t["by"], labels, fill_frames))))
    rng = np.random.default_rng(seed)
    n = len(t["bx"])
    if own_gap_share is not None:
        gaps = np.random.default_rng([seed, 3])
        missing = ~np.isfinite(t["bx"])[labels != "X"]
        moved = np.resize(np.roll(missing, gaps.integers(0, len(missing))), n)
        mask = np.zeros(n, dtype=bool)
        for a, b in zip(*_runs(moved)):
            if gaps.random() < own_gap_share:
                mask[a:b] = True
    else:
        ours = load_tracking()
        missing = ~np.isfinite(ours["bx"])[graded_mask()]
        start = rng.integers(0, len(missing) - n) if len(missing) > n else 0
        mask = np.resize(missing[start:], n)
        edges = np.diff(np.concatenate(([0], mask.astype(int), [0])))
        mask = np.zeros(n, dtype=bool)
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            if rng.random() < keep_gaps:
                mask[a:b] = True

    out = dict(t)
    cbx, cby = t["bx"], t["by"]
    if labels is not None and waiting_share:
        cbx, cby = _waiting_ball(t, labels, seed, waiting_share, **(waiting_kw or {}))
    if labels is not None and after_whistle_share:
        cbx, cby = _after_whistle_ball(cbx, cby, labels, seed, after_whistle_share, **(after_kw or {}))
    bx = _smooth_visible(cbx + rng.normal(0, ball_noise, n), ball_smooth)
    by = _smooth_visible(cby + rng.normal(0, ball_noise, n), ball_smooth)
    bx = bx + _drift(n, ball_drift, drift_tau, rng, drift_smooth)
    by = by + _drift(n, ball_drift, drift_tau, rng, drift_smooth)
    bx[mask] = np.nan
    by[mask] = np.nan
    out["bx"], out["by"] = bx, by

    px = t["px"] + _smooth_noise(rng, player_noise, t["px"].shape, player_smooth)
    py = t["py"] + _smooth_noise(rng, player_noise, t["py"].shape, player_smooth)
    for j in range(px.shape[1]):
        f = 0
        while f < n:
            gap = rng.integers(*drop_chunk)
            if rng.random() < drop_rate:
                px[f:f + gap, j] = np.nan
                py[f:f + gap, j] = np.nan
            f += gap
    out["px"], out["py"] = px, py
    out["ps"] = np.where(np.isfinite(px), t["ps"], np.nan)
    return out


# ---------------------------------------------------------------- public API

def _clean_game(name):
    """Clean tracking, labels before the control rule, and blocks of a validation match."""
    cache = label_cache(name)
    if os.path.exists(cache):
        z = np.load(cache)
        clean = {k: z[k] for k in z.files if k not in ("labels", "blocks")}
        labels, blocks = z["labels"], [tuple(b) for b in z["blocks"]]
    else:
        folder = os.path.join(ensure_downloaded(), name)
        loader = _load_epts_game if name == "Sample_Game_3" else _load_csv_game
        _, px, py, bx, by, team = loader(folder, name)
        L, W = PITCH
        px, py, bx, by = px * L, py * W, bx * L, by * W
        clean = {"px": px, "py": py, "ps": _speed(px, py), "bx": bx, "by": by, "team": team}
        labels = labels_from_events(bench_events(name, bx), len(bx))
        g = labels != "X"
        edges = np.diff(np.concatenate(([0], g.astype(int), [0])))
        blocks = [(i + 1, a, b - 1) for i, (a, b) in
                  enumerate(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))]
        np.savez_compressed(cache, labels=labels, blocks=np.array(blocks), **clean)
    return clean, labels, blocks


def load_game(name, seed=0, own_gap_share=None):
    """Degraded tracks, labels and blocks of a validation match. With `own_gap_share` (0.0 for the
    team evidence training rows) our match is not read and no gaps of it are copied in."""
    clean, labels, blocks = _clean_game(name)
    if CONTROL_SECONDS:
        labels = require_control(labels, bench_events(name, clean["bx"]), CONTROL_SECONDS)
    return degrade(clean, seed=seed, labels=labels, waiting_share=WAITING_SHARE,
                   after_whistle_share=AFTER_WHISTLE_SHARE, fill_frames=BALL_FILL_FRAMES,
                   own_gap_share=own_gap_share), labels, blocks


def load_spread(name, seed, setting):
    """Degraded tracks under the given `degrade` arguments, and labels before the control rule (dead
    frames are the same with it). With `own_gap_share` in `setting` our match is not read."""
    clean, labels, _ = _clean_game(name)
    return degrade(clean, seed=seed, labels=labels, **setting), labels


def events_for(name):
    """The event list of a validation match, frames on its tracking timeline."""
    return _events(os.path.join(ensure_downloaded(), name), name)


def summary(labels):
    g = labels[labels != "X"]
    ch = np.flatnonzero(g[1:] != g[:-1]) + 1
    st, en = np.r_[0, ch], np.r_[ch, len(g)]
    spells = [(e - s) / FPS for s, e in zip(st, en) if g[s] in "HA"]
    return {"graded_min": len(g) / FPS / 60, "D_share": float((g == "D").mean()),
            "possessions": len(spells), "mean_s": float(np.mean(spells)),
            "per_5min": len(spells) / (len(g) / FPS / 300)}


if __name__ == "__main__":
    for name in GAMES:
        t, labels, blocks = load_game(name)
        s = summary(labels)
        print("{}: {:.0f} min graded, D {:.0%}, {} possessions, mean {:.1f}s, {:.1f} per 5 min, "
              "ball visible {:.0%}".format(name, s["graded_min"], s["D_share"], s["possessions"],
                                           s["mean_s"], s["per_5min"],
                                           np.isfinite(t["bx"])[labels != "X"].mean()))
