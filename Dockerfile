# A pinned Linux environment for reproducing the submission (scripts/reproduce.sh in a container).
# On Apple silicon this image runs under emulation.
#
#   docker build --platform linux/amd64 -t mtah-possession .
#   docker run --rm --platform linux/amd64 -v /path/to/data:/data:ro -v "$PWD/out":/out mtah-possession
#
# /path/to/data holds tactical/ (as in TEAM.md). The file is written to out/submission.csv and its
# SHA256 is printed at the end; compare it with the Linux x86_64 hash in TEAM.md.
FROM --platform=linux/amd64 python:3.13-slim-bookworm@sha256:70bda1e795ded2a684ef9815955e05829b14ee5bb5c3eb2a940648e88f10fce9

RUN apt-get update \
 && apt-get install -y --no-install-recommends git curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /repo
COPY . /repo

ENV PYTHON=python3.13
ENTRYPOINT ["scripts/reproduce.sh", "/data", "/out/submission.csv"]
