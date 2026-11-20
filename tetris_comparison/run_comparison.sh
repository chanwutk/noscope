#!/bin/sh
# Run from the local experiment workspace. Keep SSH attached for rootless Docker.
set -eu
comparison=$(cd "$(dirname "$0")" && pwd)
checkout=$(cd "$comparison/.." && pwd)
workspace=$(cd "$checkout/.." && pwd)
remote=/work/cwkt/projects/noscope-released
gpu=${1:-}
case "$gpu" in
    ''|*[!0-9]*) echo 'Usage: run_comparison.sh GPU [DATASET ...]' >&2; exit 2 ;;
esac
shift
if [ "$#" -eq 0 ]; then
    set -- caldot1-y05 caldot2-y05 ams-y05 jnc0 jnc2 jnc6 jnc7
fi
for dataset do
    case "$dataset" in
        caldot1-y05|caldot2-y05|ams-y05|jnc0|jnc2|jnc6|jnc7) ;;
        *) echo "Unknown NoScope dataset: $dataset" >&2; exit 2 ;;
    esac
done

rsync -a --exclude=__pycache__ "$comparison/" "bombe:$remote/tetris_comparison/"
rsync -a "$checkout/optimizer/noscope_optimizer.py" "bombe:$remote/optimizer/"
run_stride() {
    dataset=$1
    stride=$2
    run_dir="stride-$stride-serial"
    echo "Running NoScope stride $stride on $dataset"
    ssh -n -o BatchMode=yes bombe "$remote/tetris_comparison/run_released_dataset.sh $dataset $stride $gpu"
    ssh -o BatchMode=yes ace bash -s -- "$dataset" "$stride" <<'REMOTE'
set -eu
dataset=$1
stride=$2
run_dir="stride-$stride-serial"
output="/data/cwkt/data/polyis-cache/noscope/$dataset/$run_dir"
mkdir -p "$output"
rsync -a --include=run.json --include=predictions.npz --include=plans.json --exclude='*' \
    "bombe:/data/cwkt/data/noscope-pilot/$dataset/stride-$stride/released-test-serial/" "$output/"
docker exec -w /polyis -e PYTHONPATH=/polyis polyis python -m measurements.r7o3_noscope_score \
    "$dataset" "/polyis-cache/noscope/$dataset/$run_dir"
jq -e --arg dataset "$dataset" --argjson stride "$stride" '
    .dataset == $dataset and ([.records[].target] | sort) == [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.25]
    and all(.records[]; .stride == $stride and .threshold_skip_distance == $stride)
    and .retiming.statistic == "median-of-three"
    and all(.records[]; (.timing_repeats | length) == 3)
' "$output/scores.json" >/dev/null
jq -r '.records[] | [.target,.fps,.selection.f1,.detector_frames] | @tsv' "$output/scores.json"
REMOTE
    local_output="$dataset_output/$run_dir"
    mkdir -p "$local_output"
    rsync -a "ace:/data/cwkt/data/polyis-cache/noscope/$dataset/$run_dir/" "$local_output/"
}

for dataset do
    dataset_output="$workspace/tetris/measurements-results/r7o3_noscope/$dataset"
    run_stride "$dataset" 5
    run_stride "$dataset" 15
    jq -es '
        def test_metadata: {dataset, frames, clips, detector_ms_per_frame, ground_truth_count_sha256, evaluator};
        .[0] as $short | .[1] as $long
        | if ($short|test_metadata) == ($long|test_metadata) then
            ($long|test_metadata) + {
                retiming_by_stride: {"5": $short.retiming, "15": $long.retiming},
                records: ([$short.records[], $long.records[]]
                          | group_by(.target)
                          | map(min_by([.optimizer_cost, .stride])))
            }
          else error("Stride sweeps used different test inputs") end
    ' "$dataset_output/stride-5-serial/scores.json" "$dataset_output/stride-15-serial/scores.json" \
        > "$dataset_output/scores-selected.json.tmp"
    mv "$dataset_output/scores-selected.json.tmp" "$dataset_output/scores-selected.json"
done
