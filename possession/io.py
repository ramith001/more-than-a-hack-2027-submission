"""Fast loading of the tactical ATD into numpy arrays, cached next to the data.

Arrays (video frame index 0..N-1):
    px, py, ps : (N, 46) player x, y in metres and speed in m/s, NaN when missing
    bx, by     : (N,) ball x, y in metres, NaN when missing
    bz_speed   : (N,) third value of the ball group, NaN when missing
    team       : (46,) 0 home, 1 away for each track slot
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames, half_video_blocks  # noqa: E402

PITCH = GAMES[GAME]["pitch"]
CACHE_VERSION = 1  # bump when parsing changes


def team_of_tracks(meta_path):
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([0 if p.get("teamId") == home else 1 for p in root.iter("Player")], dtype=np.int8)


def _parse(raw_path, n_frames, n_slots):
    data = np.full((n_frames, n_slots * 3 + 3), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            row = parts[1].replace(";", ",") + "," + parts[2]
            data[int(parts[0]) - 1] = np.array(row.split(","), dtype=np.float32)
    return data


def _cache_key(raw_path):
    st = os.stat(raw_path)
    return np.array([st.st_size, st.st_mtime_ns, CACHE_VERSION], dtype=np.int64)


def load_tracking(data_dir=None, use_cache=True):
    meta_path, raw_path = require(["atd_meta", "atd_raw"], GAME, data_dir)
    cache = os.path.join(os.path.dirname(raw_path), "cache_tracking.npz")
    team = team_of_tracks(meta_path)
    key = _cache_key(raw_path)
    if use_cache and os.path.exists(cache):
        with np.load(cache) as z:
            if "cache_key" in z.files and np.array_equal(z["cache_key"], key):
                return {k: z[k] for k in z.files if k != "cache_key"}

    n = source_frames("tactical", GAME)
    data = _parse(raw_path, n, len(team))
    length, width = PITCH
    players = data[:, :-3].reshape(n, len(team), 3)
    out = {
        "px": players[:, :, 0] * length,
        "py": players[:, :, 1] * width,
        "ps": players[:, :, 2],
        "bx": data[:, -3] * length,
        "by": data[:, -2] * width,
        "bz_speed": data[:, -1],
        "team": team,
    }
    np.savez_compressed(cache, cache_key=key, **out)
    return out


def graded_mask(n=None):
    n = n or source_frames("tactical", GAME)
    mask = np.zeros(n, dtype=bool)
    for _, a, b in half_video_blocks(GAME):
        mask[a:b + 1] = True
    return mask


def half_of_frame(n=None):
    n = n or source_frames("tactical", GAME)
    half = np.zeros(n, dtype=np.int8)
    for h, a, b in half_video_blocks(GAME):
        half[a:b + 1] = h
    return half


def write_submission(path, labels):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    return path


def load_public_gt():
    import pandas as pd
    p = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
    return pd.read_csv(p)["label"].to_numpy().astype("<U1")


def smart_tagging_instances():
    """[(start_frame, end_frame, code, tags)] from the Smart Tagging XML, frames on the video timeline."""
    import xml.etree.ElementTree as ET
    path = require(["smart_tagging"], GAME)[0]
    out = []
    for inst in ET.parse(path).getroot().iter("instance"):
        tags = {l.findtext("group"): l.findtext("text") for l in inst.iter("label")}
        out.append((round(float(inst.findtext("start")) * 25), round(float(inst.findtext("end")) * 25),
                    (inst.findtext("code") or "").strip().upper(), tags))
    return sorted(out, key=lambda r: r[0])


def kickoff_frames(t, window=500, spot_radius=2.0, moving=2.0, seen_only=False):
    """Video frame of every kick-off: the ball leaving the centre spot after a Smart Tagging kick-off.

    The tagged instance starts a few seconds before the kick. Where the ball is tracked on the spot
    the kick is detected directly; where it is not, the tag start plus the median delay measured on
    the detected ones is used, unless `seen_only`, which keeps only the directly detected ones.
    """
    from possession.model import ball_kinematics
    length, width = PITCH
    speed, _ = ball_kinematics(t["bx"], t["by"])
    tags = [(s, tg.get("Type", "")) for s, _, code, tg in smart_tagging_instances()
            if code == "SET PIECES" and (tg.get("Type") or "").lower().startswith("kick off")]
    found = {}
    for s, kind in tags:
        w = np.arange(s, min(s + window, len(speed) - 5))
        on_spot = np.hypot(t["bx"][w] - length / 2, t["by"][w] - width / 2) < spot_radius
        hit = np.flatnonzero(on_spot & (speed[w + 5] > moving))
        found[s] = (int(w[hit[0]]) if len(hit) else None, kind)
    delays = [k - s for s, (k, _) in found.items() if k is not None]
    delay = int(np.median(delays)) if delays else 0
    return [(k if k is not None else s + delay, kind) for s, (k, kind) in sorted(found.items())
            if k is not None or not seen_only]


def forced_alive(t, seconds=1.0, seen_only=False):
    """Intervals [(a, b)] where the ball must be live: the first `seconds` after every kick-off.

    With `seen_only`, only after kick-offs where the ball is seen leaving the spot. A kick-off timed
    from its tag alone can be well off: on our match, after the tag-timed kick-off at 62094 the
    players' mean speed stays at 0.7 to 0.9 m/s for 16 s, where after every kick-off seen from the
    ball it is over 1.5 m/s within 4 s."""
    n = int(seconds * 25)
    return [(k, k + n) for k, _ in kickoff_frames(t, seen_only=seen_only)]


def players_off(t, seconds=10):
    """Intervals [(a, b)] where fewer than half of the 22 players are on the tracked pitch, averaged
    over `seconds`. Play cannot go on without them: this is the walk-off after a final whistle and
    the trip to the bench in a hydration break. In live play the count never gets near it."""
    k = int(seconds * 25)
    n = np.isfinite(t["px"]).sum(axis=1).astype(float)
    off = np.convolve(n, np.ones(k) / k, "same") < 11
    d = np.diff(np.r_[0, off.astype(np.int8), 0])
    return [(a, b - 1) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]


def forced_dead(t):
    """Intervals [(a, b)] that must be dead ball: each graded half up to its opening kick-off, and
    every stretch with most players off the pitch (players_off).

    The frame before the detected kick is the last one forced: detection is exact where the ball is
    tracked on the spot (113742, the published second-half kick-off), and a half whose grading
    starts at the kick-off itself gets nothing.
    """
    out = []
    starts = [k for k, kind in kickoff_frames(t) if "start" in kind.lower()]
    for _, a, b in half_video_blocks(GAME):
        inside = [k for k in starts if a - 25 <= k <= b]
        if inside and inside[0] - 1 > a + 25:
            out.append((a, inside[0] - 1))
    return out + players_off(t)


def goalkeeper_slots(meta_path=None):
    """Track slots the metadata marks as goalkeepers."""
    meta_path = meta_path or require(["atd_meta"], GAME)[0]
    slots = []
    for i, p in enumerate(ET.parse(meta_path).getroot().iter("Player")):
        vals = {pp.findtext("Name"): pp.findtext("Value") for pp in p.iter("ProviderParameter")}
        if vals.get("position_type") == "Goalkeeper":
            slots.append(i)
    return slots


def clean_keepers(t, end_zone=35.0):
    """Put the two goalkeeper slots on the right team for where they stand.

    The tracker has one goalkeeper slot per team (marked in the metadata), but it sometimes
    follows the other team's keeper with it, so a slot labelled Athletic can spend minutes in
    Barcelona's goal. In each half the end each team defends is read from where its players stand
    on average, and a goalkeeper slot within `end_zone` metres of a goal line is given the team
    defending that goal. Outfield slots are never changed. Returns a copy of t with a per-frame
    team array "team_frame".
    """
    length, _ = PITCH
    px, team = t["px"], t["team"]
    n = len(px)
    team_frame = np.repeat(team[None, :].astype(np.int8), n, axis=0)
    half = half_of_frame(n)
    for s in goalkeeper_slots():
        x = px[:, s]
        for h in (1, 2):
            m = half == h
            if not m.any():
                continue
            home_left = np.nanmean(px[m][:, team == 0]) < np.nanmean(px[m][:, team == 1])
            left_defender = 0 if home_left else 1
            team_frame[m & (x < end_zone), s] = left_defender
            team_frame[m & (x > length - end_zone), s] = 1 - left_defender
    return dict(t, team_frame=team_frame)
