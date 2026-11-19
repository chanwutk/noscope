#!/bin/sh
# Run the released cascade on the held-out video with validation-selected plans.
set -eu
comparison=$(cd "$(dirname "$0")" && pwd)
source_dir=$(cd "$1" && pwd)
data=$(cd "$2" && pwd)
gpu=$3
output="$data/released-test"
mkdir -p "$output"
frames=$(cat "$data/released-inputs/test.frames")
stride=$NOSCOPE_STRIDE
dump="test-stride-$stride.bin"
# Avoid the released loader's allocation past the end of an incomplete stride.
native_frames=$((frames / stride * stride))
expected_dump=$((native_frames / stride * (608 * 608 * 3 + 100 * 100 * 3 + 50 * 50 * 3 * 4)))
if [ -f "$output/$dump" ]; then
    [ "$(stat -c %s "$output/$dump")" -eq "$expected_dump" ]
fi
collect() {
    docker run --rm --entrypoint python -v "$comparison:/work:ro" -v "$data:/inputs:ro" \
        -v "$output:/output" polyis-polyis /work/released_results.py "$1" /inputs /output
}

collect plans
jq -e --argjson stride "$stride" 'length > 0 and all(.[]; .stride == $stride)' "$output/plans.json" >/dev/null
jq -r 'to_entries[] | [.key, .value.graph, .value.threshold_diff, .value.threshold_lower_cnn, .value.threshold_upper_cnn] | @tsv' \
    "$output/plans.json" > "$output/plans.tsv"
while read -r index graph diff lower upper; do
    if [ -s "$output/test-$index.csv" ]; then
        continue
    fi
    docker run --rm --gpus all -e "CUDA_VISIBLE_DEVICES=$gpu" \
        -e CUDA_CACHE_PATH=/cuda-cache -e CUDA_CACHE_MAXSIZE=2147483648 \
        -v "$source_dir:/speedhax:ro" -v "$data/released-build:/build:ro" \
        -v "$data:/inputs:ro" -v "$output:/output" -v "$data/released-cuda-cache:/cuda-cache" \
        --entrypoint /build/noscope noscope-released-inference \
        --replay_detector=1 --diff_thresh="$diff" --distill_thresh_lower="$lower" \
        --distill_thresh_upper="$upper" --skip_small_cnn=0 --skip_diff_detection=0 \
        --skip="$stride" --avg_fname="/inputs/${graph%/*}/avg.txt" --graph="/inputs/$graph" \
        --video=/inputs/released-inputs/test.mp4 --start_from=0 --nb_frames="$native_frames" \
        --confidence_csv="/output/test-$index.tmp" --dumped_videos="/output/$dump" \
        --diff_detection_weights=/dev/null --use_blocked=0 --ref_image=0 > "$output/test-$index.log" 2>&1
    [ "$(wc -l < "$output/test-$index.tmp")" -eq "$((native_frames + 3))" ]
    [ "$(stat -c %s "$output/$dump")" -eq "$expected_dump" ]
    mv "$output/test-$index.tmp" "$output/test-$index.csv"
    echo "Finished test operating point $index"
done < "$output/plans.tsv"
collect collect
