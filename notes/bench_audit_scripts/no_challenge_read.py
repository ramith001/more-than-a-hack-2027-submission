"""Keep bench runs from reading the challenge match. Import this BEFORE any possession module.

    import no_challenge_read as N      # installs the audit guard
    from possession import external    # (or import dfl_recheck, which does this)
    N.install()                        # stubs external.load_tracking and external.graded_mask

The guard raises RuntimeError on any open, os.listdir, os.scandir or glob event whose path contains
BAR-ATH, data/tactical, data/broadcast or Sample_Game_3.

external.degrade() normally reads the challenge tracking and copies a share of its missing-ball gaps
into the bench. The stubs give it one non-missing frame, so no gaps are copied and no random number is
drawn for the start. That is the same as keep_gaps 0 but with a different random stream than earlier
bench numbers, so absolute scores are not comparable with earlier numbers; gains are paired
against the defaults on the same bench.
"""
import sys

import numpy as np

BLOCKED = ("BAR-ATH", "data/tactical", "data/broadcast", "Sample_Game_3")
EVENTS = ("open", "os.listdir", "os.scandir", "glob.glob")


def _hook(event, args):
    if event in EVENTS and args:
        path = args[0]
        if isinstance(path, bytes):
            path = path.decode("utf-8", "replace")
        if isinstance(path, str) and any(b in path for b in BLOCKED):
            raise RuntimeError("blocked path: {!r}".format(path))


sys.addaudithook(_hook)


def install():
    from possession import external
    external.load_tracking = lambda *a, **k: {"bx": np.zeros(1)}
    external.graded_mask = lambda *a, **k: np.array([True])
