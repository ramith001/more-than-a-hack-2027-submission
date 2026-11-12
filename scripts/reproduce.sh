#!/usr/bin/env bash
# Regenerate the submission file from the match data, byte-identically, with one command.
#   scripts/reproduce.sh [DATA_DIR] [OUT]
# DATA_DIR (or env DATA_DIR) holds tactical/ (and optionally models/, broadcast/, external/).
# OUT defaults to evaluation/results/aditya_reproduce.csv.
# Env: PYTHON (default python3.13), VENV (default .venv).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${1:-${DATA_DIR:-}}"
OUT="${2:-evaluation/results/aditya_reproduce.csv}"
PYTHON="${PYTHON:-python3.13}"
VENV="${VENV:-.venv}"
case "$VENV" in /*) ;; *) VENV="$ROOT/$VENV" ;; esac

die() { echo "reproduce.sh: $*" >&2; exit 1; }

[ -n "$DATA_DIR" ] || die "usage: scripts/reproduce.sh DATA_DIR [OUT]  (or set DATA_DIR)"
DATA_DIR="$(cd "$DATA_DIR" 2>/dev/null && pwd)" || die "DATA_DIR does not exist"
[ -d "$DATA_DIR/tactical" ] || die "$DATA_DIR/tactical not found"
case "$OUT" in *v2.csv) die "refusing to write v2.csv" ;; esac

# portable sha256 of one file
sha256() {
  if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

# link TARGET PATH: create a symlink; fine if it already points there; stop if anything else is in the way.
link() {
  local target="$1" path="$2"
  if [ "$path" -ef "$target" ]; then
    return 0   # already the same file (e.g. a checkout with the data unpacked in data/)
  elif [ -L "$path" ]; then
    [ "$(readlink "$path")" = "$target" ] || die "$path already exists and points elsewhere ($(readlink "$path")); not overwriting"
  elif [ -e "$path" ]; then
    die "$path already exists and is not a link to $target; not overwriting"
  else
    ln -s "$target" "$path"
  fi
}

# --- data layout ---------------------------------------------------------------------------
# data/tactical is a real, git-ignored directory in the repo holding one symlink per file of
# DATA_DIR/tactical, so nothing is written into DATA_DIR. Caches the code writes
# (cache_tracking.npz) land in data/tactical in the repo, not in DATA_DIR; caches
# found in DATA_DIR are not linked, so a stale one cannot be reused or written through.
[ ! -L data/tactical ] || die "data/tactical is a symlink; replace it with a real directory"
mkdir -p data/tactical
for f in "$DATA_DIR"/tactical/*; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in cache_*) continue ;; esac
  link "$f" "data/tactical/$(basename "$f")"
done
for need in BAR-ATH_tactical_atd.txt BAR-ATH_tactical_atd.xml BAR-ATH_tactical_homography.csv BAR-ATH_tactical_smart_tagging.xml; do
  [ -e "data/tactical/$need" ] || die "missing $need in $DATA_DIR/tactical"
done
# data/models and data/external are filled by the pinned downloads below and in external.py.

# --- python environment --------------------------------------------------------------------
command -v "$PYTHON" >/dev/null || die "$PYTHON not found; install Python 3.13 or set PYTHON"
[ -f requirements-model.txt ] || die "requirements-model.txt missing"
[ -x "$VENV/bin/python" ] || "$PYTHON" -m venv "$VENV"
PYV="$("$VENV/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
[ "$PYV" = "3.13" ] || die "$VENV has Python $PYV, need 3.13 (set VENV to a fresh path)"
"$VENV/bin/python" -m pip install --quiet --only-binary :all: -r requirements-model.txt

export PATH="$VENV/bin:$PATH"

# --- determinism: one thread everywhere, fixed hash seed, one data root ----------------------------
export MTAH_DATA_DIR="$ROOT/data"   # games.py prefers this over ./data
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONHASHSEED=0

# --- run -----------------------------------------------------------------------------------
mkdir -p "$(dirname "$OUT")"
"$VENV/bin/python" possession/run.py "$OUT"
"$VENV/bin/python" evaluation/validate_submission.py "$OUT"
echo "SHA256 $(sha256 "$OUT")  $OUT"
