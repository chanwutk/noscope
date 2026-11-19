#!/bin/sh
# Pass the physical indices of GPUs reserved for this run.
set -eu
cd "$(dirname "$0")/.."
root=$PWD
dataset=${NOSCOPE_DATASET:-caldot1-y05}
data=/data/cwkt/data/noscope-pilot/$dataset
workers=$#
[ "$workers" -ge 1 ] && [ "$workers" -le 12 ]
shard=0
for gpu do
    output="$data/compat-gpu-shards/gpu-$gpu"
    mkdir -p "$output"
    docker run -d --name "noscope-${dataset%-y05}-compat-gpu-shards-$gpu" --gpus all \
        -e "CUDA_VISIBLE_DEVICES=$gpu" -e "NOSCOPE_SHARD=$shard" \
        -e "NOSCOPE_WORKERS=$workers" -e "NOSCOPE_DATASET=$dataset" -e OPENBLAS_NUM_THREADS=1 \
        -v "$root:/noscope:ro" -v "$data/released-inputs:/inputs:ro" \
        -v "$output:/models" noscope-compat-train-gpu:official sh -ec '
            python tetris_comparison/train_shard.py --avg_fname /models/avg.npy \
                --csv_in /inputs/train.csv --video_in /inputs/train.mp4 \
                --output_dir /models --base_name "$NOSCOPE_DATASET" --objects car \
                --num_frames "$(cat /inputs/train.frames)" --start_frame 0
            python exp/to_avg.py --npy_fname /models/avg.npy --txt_fname /models/avg.txt
        '
    shard=$((shard + 1))
done
