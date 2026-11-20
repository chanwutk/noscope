#!/bin/sh
# Run the released NoScope optimizer on each completed validation profile.
set -eu
checkout=$(cd "$(dirname "$0")/.." && pwd)
data=$(cd "$1" && pwd)
stride=$NOSCOPE_STRIDE
mkdir -p "$data/released-calibration"

calibrate() {
    name=$1
    output="$data/released-calibration/$name.json"
    [ ! -s "$output" ] || return 0
    docker run --rm -e OPENBLAS_NUM_THREADS=1 -v "$checkout:/noscope:ro" \
        -v "$data:/data:ro" noscope-legacy-train python -c '
import json
import sys

stride = int(sys.argv[2])
sys.path.insert(0, "/noscope/optimizer")
import noscope_optimizer as optimizer

optimizer.POOL_SIZE = 4
optimizer.SKIP_DISTANCES = (stride,)
model = sys.argv[1]
frames = int(open("/data/released-inputs/valid.frames").read())
frames = frames // 30 * 30
reference = "/data/released-inputs/valid.csv"
profile = "/data/released-validation/{}.csv".format(model)
results = []
# Default sweep of paired FP/FN targets (TARGET_ERROR_RATES in optimizer/noscope_motherdog.py).
for target in (0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0):
    plan = optimizer.main("car", reference, profile, target, target, 0, frames)
    plan["target"] = target
    results.append(plan)
print(json.dumps(results, default=lambda value: value.item(), indent=2))
' "$name" "$stride" > "$output.tmp" 2> "$data/released-calibration/$name.log" || return 1
    mv "$output.tmp" "$output"
    echo "Calibrated $name"
}

pids=
for profile in "$data"/released-validation/convnet_*.csv; do
    name=${profile##*/}
    calibrate "${name%.csv}" &
    pids="$pids $!"
done
status=0
for pid in $pids; do
    wait "$pid" || status=1
done
exit "$status"
