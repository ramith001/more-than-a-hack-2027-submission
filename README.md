# More than a Hack 2027, Phase 1: Ball Possession

Our submission for the ball possession challenge by Metrica Sports and FC Barcelona. For every
frame of the BAR-ATH match the pipeline writes one label: H (home in possession), A (away in
possession), D (dead ball) or X (abstain). Entries are ranked on spellF1.

Team: Ramith, Aslan, Aditya.

## What is in this repository

- `possession/` the model and the pipeline (`run.py` is the entry point)
- `evaluation/` the organisers' scorer and the public 5 minute ground truth
- `docs/technical_document.md` the technical document
- `docs/results_report.md` every score and check in one place
- `notes/evidence/` the scripts behind the claims in the technical document; `team_evidence_all_five.py` with its `.json` and `.log` is the test behind the results table
- `notes/bench_audit_scripts/` the bench that test runs on
- `TEAM.md` the team guide
- `scripts/reproduce.sh` one command that rebuilds the submission file
- `Dockerfile` a pinned Linux environment for the same command
- `docs/ORGANISERS_README.md` the organisers' own README, unchanged. It sits here because this file took its place at the top level. The organisers' other files (`OVERVIEW.md`, `CHALLENGE.md`, `RULES.md`, `FAQ.md`, `DATA.md`, `EVALUATION.md`, `SUBMISSION.md`) are their current versions, unchanged

The match data is not part of this repository and is never committed.

## Setup

You need Python 3.13 and the match data folder as the organisers provide it. The data folder must
contain a `tactical/` directory with these files:

```
BAR-ATH_tactical_atd.txt
BAR-ATH_tactical_atd.xml
BAR-ATH_tactical_homography.csv
BAR-ATH_tactical_smart_tagging.xml
```

Keep the data outside this repository folder. `data/` is ignored by git. The first run also
downloads Metrica's public sample matches and the DFL open data into `data/external/`.

## How to run

```bash
scripts/reproduce.sh /path/to/data out.csv
```

The script creates a `.venv` from `requirements-model.txt`, fixes the thread counts and the hash
seed, runs `possession/run.py`, validates the file with `evaluation/validate_submission.py` and
prints the SHA256 of the output. Without the second argument the file goes to
`evaluation/results/aditya_reproduce.csv`.

With Docker:

```bash
docker build --platform linux/amd64 -t mtah-possession .
docker run --rm --platform linux/amd64 -v /path/to/data:/data:ro -v "$PWD/out":/out mtah-possession
```

The file is written to `out/submission.csv`.

## Expected output

```
SHA256 046398fafb81c8a752c9853a8848d997301679437d1817a172244c1a9b043386
```

## Other commands

```bash
python possession/validate.py                  # score the model on the validation bench
python evaluation/run_evaluation.py            # score files in evaluation/results/ on the public 5 minutes
python notes/evidence/game3_holdout.py         # check run on Metrica games 1 and 2 only
```

`TEAM.md` has the details of the data layout, the validation bench and the rules we follow.
