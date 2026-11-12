"""Validation matches built from the DFL (Sportec) open data, same interface as possession/external.py.

Seven Bundesliga matches with raw tracking at 25 fps and event data. The XML files are big and stay
outside the repo: they are downloaded from Figshare into data/external/dfl (git-ignored) and checked
against their published MD5, or read from MTAH_DFL_DIR if that is set. The clean arrays are cached there,
in a cache/ subfolder, as compressed npz (no pickle). Nothing from the files is ever executed.

    MTAH_DFL_DIR=/path python possession/external_dfl.py    # builds the caches, prints a summary

The timeline is the first half, a gap of GAP X frames, then the second half, so nothing bridges
half time. Labels come from the tracking's own ball status, then the same control rule and the
same degradation as the Metrica bench are applied.

The ball is blanked on all BallStatus=0 frames, because DFL freezes the ball position while the ball
is dead. The Metrica clean ball is missing on 89-96% of dead frames. This makes dead ball slightly
easier here than on Metrica.
"""

import glob
import hashlib
import os
import sys
import urllib.request
from datetime import datetime

import numpy as np
import xml.etree.ElementTree as ET

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import external  # noqa: E402

GAMES = ("J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9")
GAP = 125            # X frames between the halves
NEAR = 10.0          # seconds: an event this close to a half is moved onto it, farther ones are dropped
SET_PIECES = {"KickOff": "KICK OFF", "ThrowIn": "THROW IN", "FreeKick": "FREE KICK",
              "CornerKick": "CORNER KICK", "GoalKick": "GOAL KICK", "Penalty": "PENALTY"}
COMPLETED = ("successfullyCompleted", "successful")


# The 21 files used, as published on Figshare (doi 10.6084/m9.figshare.28196177, CC BY 4.0):
# name, Figshare file id, MD5.
FIGSHARE = "https://ndownloader.figshare.com/files/{}"
FILES = (
    ("DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WN1.xml", 51643472, "37b0f85588c0acfb995fe633c7ac2e0d"),
    ("DFL_02_01_matchinformation_DFL-COM-000001_DFL-MAT-J03WMX.xml", 51643475, "bc5d5b21dbcf36aa30f4cd0428309f6b"),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml", 51643478, "b9710781d400dd812f62b0a6e3d7bce8"),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOY.xml", 51643481, "7b021e577fca917a63111d55cc29ae5b"),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WQQ.xml", 51643484, "5979209af316a914ce57347d0633df6c"),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WPY.xml", 51643487, "65a4568e4f374af94d3c1015cbc7c4f5"),
    ("DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WR9.xml", 51643490, "5e2d580d3d8ed97e32959de90d04992b"),
    ("DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WMX.xml", 51643493, "2d88e180a1f63c7e4157feb980fc7255"),
    ("DFL_03_02_events_raw_DFL-COM-000001_DFL-MAT-J03WN1.xml", 51643496, "83e8ed7ea8e8ebbbdc61f61ac3534008"),
    ("DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml", 51643499, "ab66e51512f39d06f49f27dd3b4c2f40"),
    ("DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOY.xml", 51643502, "64b1be01d6ff6848168efebd56f344b9"),
    ("DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WPY.xml", 51643505, "f68482441d38723d69f244f896482803"),
    ("DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WQQ.xml", 51643508, "b7e9d82af4034539d802319be6c57776"),
    ("DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WR9.xml", 51643511, "c13bc5efa5eca4a022cc678210b87d5f"),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WMX.xml", 51643514, "e4273a2adf5dbd9d4ec42def2f01905c"),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000001_DFL-MAT-J03WN1.xml", 51643517, "bacb2f609c1d1b0bfc906f6b9d4109f2"),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml", 51643520, "3efff25a86a61768e7d253b4bba137a3"),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOY.xml", 51643523, "4e1063e0a0ae499920cf68ac1aa3ab4e"),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WPY.xml", 51643526, "e56ff1336a334f8da51110fbe7cab764"),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WQQ.xml", 51643529, "9105bad987191d3c3c362c1fce9a4fcd"),
    ("DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WR9.xml", 51643532, "e060bb7d7a5a3bfbd15bbf9bc631024d"),
)


def _md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_downloaded(d):
    """Download any missing file into `d` and check every file against its published MD5."""
    os.makedirs(d, exist_ok=True)
    for name, fid, md5 in FILES:
        path = os.path.join(d, name)
        if not os.path.exists(path):
            print("downloading", name, file=sys.stderr, flush=True)
            tmp = path + ".partial"
            urllib.request.urlretrieve(FIGSHARE.format(fid), tmp)
            os.replace(tmp, path)
        got = _md5(path)
        if got != md5:
            raise SystemExit("{}: MD5 {} does not match the published {}".format(path, got, md5))


_CHECKED = []


def data_dir():
    """MTAH_DFL_DIR if set, otherwise data/external/dfl in the repo; files fetched and checked once."""
    d = os.environ.get("MTAH_DFL_DIR") or os.path.join(external.EXT_DIR, "dfl")
    if not _CHECKED:
        ensure_downloaded(d)
        _CHECKED.append(d)
    return d


def _file(kind, name):
    found = glob.glob(os.path.join(data_dir(), "DFL_{}_*_DFL-MAT-{}.xml".format(kind, name)))
    if len(found) != 1:
        raise FileNotFoundError("expected one DFL_{}_*{}.xml in {}".format(kind, name, data_dir()))
    return found[0]


def _teams(name):
    """Team id -> 0 (home) or 1 (away), from the Role attribute."""
    root = ET.parse(_file("02_01_matchinformation", name)).getroot()
    env = root.find("MatchInformation/Environment")
    assert (float(env.get("PitchX")), float(env.get("PitchY"))) == external.PITCH, \
        "{}: pitch is {} x {}, not {}".format(name, env.get("PitchX"), env.get("PitchY"), external.PITCH)
    return {t.get("TeamId"): {"home": 0, "guest": 1}[t.get("Role")] for t in root.iter("Team")}


def build(name):
    """Clean arrays of one match: the cache content."""
    teams = _teams(name)
    halves, times, buf = {}, {}, []
    for _, e in ET.iterparse(_file("04_03_positions_raw_observed", name)):
        if e.tag == "Frame":
            a = e.attrib
            buf.append((a["N"], a["X"], a["Y"], a.get("BallPossession"), a.get("BallStatus"), a["T"]))
            e.clear()
        elif e.tag == "FrameSet":
            who, sec = e.get("TeamId"), e.get("GameSection")
            if who != "referee":
                n, x, y, bp, bs, tt = zip(*buf)
                half = halves.setdefault(sec, {"players": {}})
                n, x, y = np.array(n, dtype=np.int64), np.array(x, dtype=float), np.array(y, dtype=float)
                if who == "BALL":
                    half["ball"] = (n, x, y, bp, bs)
                    times[sec] = datetime.fromisoformat(tt[0])
                else:
                    half["players"][e.get("PersonId")] = (teams[who], n, x, y)
            buf = []
            e.clear()
    first, second = halves["firstHalf"], halves["secondHalf"]
    lens = []
    for h in (first, second):
        n = h["ball"][0]
        assert (np.diff(n) == 1).all(), name + ": gaps in the ball frame numbers"
        lens.append(len(n))
    total = lens[0] + GAP + lens[1]
    offs = (0, lens[0] + GAP)

    people = {}
    for h in (first, second):
        for pid, (tm, *_) in h["players"].items():
            people[pid] = tm
    cols = sorted(people, key=lambda p: (people[p], p))
    px = np.full((total, len(cols)), np.nan)
    py = np.full((total, len(cols)), np.nan)
    bx, by = np.full(total, np.nan), np.full(total, np.nan)
    lab = np.full(total, "X", dtype="<U1")
    L, W = external.PITCH
    for h, off in zip((first, second), offs):
        n0 = h["ball"][0][0]
        sl = slice(off, off + len(h["ball"][0]))
        n, x, y, bp, bs = h["ball"]
        bp, bs = np.array(bp, dtype=np.int8), np.array(bs, dtype=np.int8)
        assert set(np.unique(bs)) <= {0, 1} and set(np.unique(bp[bs == 1])) <= {1, 2}, \
            name + ": unexpected BallStatus or BallPossession values"
        # the ball position is frozen while BallStatus is 0; the Metrica clean ball is mostly missing
        # in dead time and the degradation assumes that, so drop it there
        live = bs == 1
        bx[sl] = np.where(live, x + L / 2, np.nan)
        by[sl] = np.where(live, y + W / 2, np.nan)
        # BallPossession 1 = home, 2 = away (inferred from the data); dead comes first
        lab[sl] = np.where(~live, "D", np.where(bp == 1, "H", "A"))
        for j, pid in enumerate(cols):
            if pid in h["players"]:
                _, pn, x, y = h["players"][pid]
                i = pn - n0
                ok = (i >= 0) & (i < len(n))
                px[off + i[ok], j] = x[ok] + L / 2
                py[off + i[ok], j] = y[ok] + W / 2
    team = np.array([people[p] for p in cols], dtype=np.int8)
    blocks = np.array([(1, 0, lens[0] - 1), (2, offs[1], total - 1)])
    t0 = np.array([times["firstHalf"].timestamp(), times["secondHalf"].timestamp()])
    return dict(px=px, py=py, ps=external._speed(px, py), bx=bx, by=by, team=team, labels=lab,
                blocks=blocks, half_t0=t0, half_n=np.array(lens))


def _clean(name):
    cache = os.path.join(data_dir(), "cache", name + ".npz")
    if not os.path.exists(cache):
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, **build(name))
    z = np.load(cache)    # allow_pickle stays False
    return {k: z[k] for k in z.files}


def raw_labels(name):
    """H/A/D/X per frame before the control rule."""
    return _clean(name)["labels"]


def _frame(inst, t0, lens):
    """Frame index of an instant on the two-half timeline, or None if it is not near either half."""
    offs = (0, lens[0] + GAP)
    best = None
    for h in (0, 1):
        k = round((inst - t0[h]) * external.FPS)
        away = max(-k, k - (lens[h] - 1), 0) / external.FPS
        if best is None or away < best[0]:
            best = away, h, k
    away, h, k = best
    if away > NEAR:
        return None
    return h + 1, offs[h] + min(max(k, 0), lens[h] - 1)


def _event(e, t0, lens, teams, skipped):
    """The list entries of one <Event> (none if it is skipped)."""
    kids = list(e)
    c = kids[0] if kids else None
    if c is None or c.tag == "Delete":
        skipped["delete"] = skipped.get("delete", 0) + 1
        return []
    where = _frame(datetime.fromisoformat(e.get("EventTime")).timestamp(), t0, lens)
    if where is None:
        skipped["dropped"] = skipped.get("dropped", 0) + 1
        return []
    period, f = where
    out = []

    def add(team, typ, sub):
        out.append(dict(team=teams[team], type=typ, subtype=sub, period=period, start=f, end=f))

    plays = [c] if c.tag == "Play" else c.findall("Play") if c.tag in SET_PIECES else []
    if c.tag == "KickOff":
        team = plays[0].get("Team") if plays else None
    else:
        team = c.get("Team")
    if team not in teams:
        skipped["no team"] = skipped.get("no team", 0) + 1
    elif c.tag in SET_PIECES:
        add(team, "SET PIECE", SET_PIECES[c.tag])
    else:
        add(team, c.tag, "")
    for p in plays:
        if p.get("Evaluation") in COMPLETED and (p.find("Pass") is not None or p.find("Cross") is not None):
            add(p.get("Team"), "PASS", "")
    return out


def events_for(name, counts=None):
    """The event list of a match, frames on its timeline, in the format of external.events_for.
    `counts`, if given, gets the number of events skipped for each reason."""
    z = _clean(name)
    skipped = {} if counts is None else counts
    teams = _teams(name)
    out = []
    for _, e in ET.iterparse(_file("03_02_events_raw", name)):
        if e.tag == "Event":
            out += _event(e, z["half_t0"], z["half_n"], teams, skipped)
            e.clear()
    return sorted(out, key=lambda x: x["start"])


def load_game(name, seed=0, control=None, own_gap_share=None):
    z = _clean(name)
    if control is None:
        control = external.CONTROL_SECONDS
    labels = external.require_control(z["labels"], events_for(name), control)
    clean = {k: z[k] for k in ("px", "py", "ps", "bx", "by", "team")}
    blocks = [tuple(int(v) for v in b) for b in z["blocks"]]
    return external.degrade(clean, seed=seed, labels=labels, waiting_share=external.WAITING_SHARE,
                            after_whistle_share=external.AFTER_WHISTLE_SHARE,
                            fill_frames=external.BALL_FILL_FRAMES, own_gap_share=own_gap_share), labels, blocks


if __name__ == "__main__":
    for name in GAMES:
        t, labels, blocks = load_game(name)
        s = external.summary(labels)
        print("{}: {:.0f} min graded, D {:.0%}, {} possessions, mean {:.1f}s, ball visible {:.0%}".format(
            name, s["graded_min"], s["D_share"], s["possessions"], s["mean_s"],
            np.isfinite(t["bx"])[labels != "X"].mean()))
