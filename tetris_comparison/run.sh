#!/bin/sh
# Prepare inputs on bombe using the Tetris image and rootless Docker service.
set -eu
cd "$(dirname "$0")"
data=/data/cwkt/data/noscope-pilot
i=0
until docker info >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -le 30 ] || { echo "Docker daemon not reachable" >&2; exit 1; }
    sleep 1
done
docker run --rm --entrypoint python \
    -v "$PWD:/work:ro" -v "$data:/data" -w /work polyis-polyis "$@"
