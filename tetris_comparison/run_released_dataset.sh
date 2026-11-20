#!/bin/sh
# Run validation and inference on bombe for one dataset with staged graphs and inputs.
set -eu
comparison=$(cd "$(dirname "$0")" && pwd)
checkout=$(cd "$comparison/.." && pwd)
dataset=$1
gpu=$3
base=/data/cwkt/data/noscope-pilot/$dataset
export NOSCOPE_STRIDE=$2
case "$NOSCOPE_STRIDE" in
    5|15) ;;
    *) echo "Unsupported NoScope stride: $NOSCOPE_STRIDE" >&2; exit 2 ;;
esac
data="$base/stride-$NOSCOPE_STRIDE"

idle_gpus() {
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits |
        awk -F, '$2+0 < 100 && $3+0 == 0 {print $1}'
}

i=0
until docker info >/dev/null 2>&1; do
    i=$((i + 1))
    [ "$i" -le 25 ] || exit 1
    sleep 1
done
[ -s "$base/released-inputs/valid.frames" ]
[ -s "$base/released-inputs/test.frames" ]
[ -x "$base/released-build/noscope" ]
[ "$(find "$base/compat-gpu-shards" -name '*.pb' -size +0c -printf '%f\n' | sort -u | wc -l)" -eq 12 ]

mkdir -p "$data"
exec 9>"$data/.lock"
flock -n 9 || { echo "$dataset stride-$NOSCOPE_STRIDE run is already active" >&2; exit 1; }
for item in released-inputs compat-gpu-shards; do
    [ -e "$data/$item" ] || cp -al "$base/$item" "$data/"
done
for item in released-build released-cuda-cache; do
    [ -e "$data/$item" ] || ln -s "$base/$item" "$data/$item"
done
[ -e "$data/seiden_manifest.json" ] || ln "$base/seiden_manifest.json" "$data/seiden_manifest.json"

"$comparison/profile_released.sh" "$checkout/speedhax" "$data" $(idle_gpus)
[ "$(find "$data/released-validation" -name 'convnet_*.csv' | wc -l)" -eq 12 ]
"$comparison/calibrate_released.sh" "$data"
[ "$(find "$data/released-calibration" -name 'convnet_*.json' | wc -l)" -eq 12 ]

output="$data/released-test-serial"
mkdir -p "$output"
device=$(nvidia-smi -i "$gpu" --query-gpu=uuid,name --format=csv,noheader)
[ -n "$device" ]
if [ -e "$output/device.txt" ]; then
    original_device=$(cat "$output/device.txt")
    [ "${original_device#*, }" = "${device#*, }" ] ||
        { echo "Use the original GPU model to resume $dataset" >&2; exit 1; }
else
    printf '%s\n' "$device" > "$output/device.txt"
fi
for repeat in 1 2 3; do
    echo "$dataset stride $NOSCOPE_STRIDE repeat $repeat of 3"
    "$comparison/test_released.sh" "$checkout/speedhax" "$data" "$gpu" "$output/repeat-$repeat" "$device"
done
docker run --rm --entrypoint python -v "$comparison:/work:ro" -v "$data:/inputs:ro" \
    -v "$output:/output" polyis-polyis /work/released_results.py retime /inputs /output
