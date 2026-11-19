#!/bin/sh
# Collect the released example's global-MSE and CNN scores on validation.
set -eu
source_dir=$(cd "$1" && pwd)
data=$(cd "$2" && pwd)
shift 2
stride=$NOSCOPE_STRIDE
mkdir -p "$data/released-validation" "$data/released-cuda-cache"

worker() {
    gpu=$1
    slot=$2
    workers=$3
    index=0
    for path in "$data"/compat-gpu-shards/gpu-*/*.pb; do
        name=${path##*/}
        name=${name%.pb}
        name=${name#*_}
        output="$data/released-validation/$name.csv"
        case " $pending " in
            *" $name "*) ;;
            *) continue ;;
        esac
        if [ "$((index % workers))" -eq "$slot" ]; then
            graph=${path#"$data"/}
            average=${graph%/*}/avg.txt
            docker run --rm --gpus all -e "CUDA_VISIBLE_DEVICES=$gpu" --label "noscope.profile=$name" \
                -e CUDA_CACHE_PATH=/cuda-cache -e CUDA_CACHE_MAXSIZE=2147483648 \
                -v "$source_dir:/speedhax:ro" -v "$data/released-build:/build:ro" \
                -v "$data:/inputs:ro" -v "$data/released-validation:/output" \
                -v "$data/released-cuda-cache:/cuda-cache" \
                --entrypoint /build/noscope noscope-released-inference \
                --replay_detector=1 --diff_thresh=0 --distill_thresh_lower=0 \
                --distill_thresh_upper=0 --skip_small_cnn=0 --skip_diff_detection=0 \
                --skip="$stride" --avg_fname="/inputs/$average" --graph="/inputs/$graph" \
                --video=/inputs/released-inputs/valid.mp4 --start_from=0 \
                --nb_frames="$frames" \
                --confidence_csv="/output/$name.tmp" --dumped_videos=/dev/null \
                --diff_detection_weights=/dev/null --use_blocked=0 --ref_image=0 \
                > "$data/released-validation/$name.log" 2>&1 || return 1
            [ "$(wc -l < "$data/released-validation/$name.tmp")" -eq "$((frames + 3))" ] || return 1
            mv "$data/released-validation/$name.tmp" "$output" || return 1
            echo "GPU $gpu validated $name"
        fi
        index=$((index + 1))
    done
}

# Snapshot pending candidates before any workers can finish and change the assignment.
pending=
for path in "$data"/compat-gpu-shards/gpu-*/*.pb; do
    name=${path##*/}
    name=${name%.pb}
    name=${name#*_}
    [ -s "$data/released-validation/$name.csv" ] || pending="$pending $name"
done

# The released loader allocates complete strides. The optimizer uses 30-frame windows.
frames=$(cat "$data/released-inputs/valid.frames")
frames=$((frames / 30 * 30))
[ -n "$pending" ] || exit 0
[ "$#" -gt 0 ]
workers=$#
slot=0
pids=
for gpu do
    worker "$gpu" "$slot" "$workers" &
    pids="$pids $!"
    slot=$((slot + 1))
done
status=0
for pid in $pids; do
    wait "$pid" || status=1
done
exit "$status"
