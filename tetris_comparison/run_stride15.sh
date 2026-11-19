#!/bin/sh
# Run from the local experiment workspace. Keep SSH attached for rootless Docker.
set -eu
comparison=$(cd "$(dirname "$0")" && pwd)
checkout=$(cd "$comparison/.." && pwd)
workspace=$(cd "$checkout/.." && pwd)
remote=/work/cwkt/projects/noscope-released

if [ "$#" -eq 0 ]; then
    set -- caldot1-y05 caldot2-y05 ams-y05
fi
for dataset do
    case "$dataset" in
        caldot1-y05|caldot2-y05|ams-y05) ;;
        *) echo "Unknown NoScope dataset: $dataset" >&2; exit 2 ;;
    esac
done

rsync -a --exclude=__pycache__ "$comparison/" "bombe:$remote/tetris_comparison/"
rsync -a "$checkout/optimizer/noscope_optimizer.py" "bombe:$remote/optimizer/"
for dataset do
    echo "Running NoScope stride 15 on $dataset"
    ssh -n -o BatchMode=yes bombe "$remote/tetris_comparison/run_released_dataset.sh $dataset"
    ssh -o BatchMode=yes ace bash -s -- "$dataset" <<'REMOTE'
set -eu
dataset=$1
output="/data/cwkt/data/polyis-cache/noscope/$dataset/stride-15"
mkdir -p "$output"
rsync -a --include=run.json --include=predictions.npz --include=plans.json --exclude='*' \
    "bombe:/data/cwkt/data/noscope-pilot/$dataset/stride-15/released-test/" "$output/"
docker exec -w /polyis -e PYTHONPATH=/polyis polyis python -m measurements.r7o3_noscope_score \
    "$dataset" "/polyis-cache/noscope/$dataset/stride-15"
jq -e --arg dataset "$dataset" '
    .dataset == $dataset and (.records | length == 3)
    and all(.records[]; .stride == 15 and .threshold_skip_distance == 15)
' "$output/scores.json" >/dev/null
jq -r '.records[] | [.target,.fps,.selection.f1,.detector_frames] | @tsv' "$output/scores.json"
REMOTE
    local_output="$workspace/tetris/measurements-results/r7o3_noscope/$dataset/stride-15"
    mkdir -p "$local_output"
    rsync -a "ace:/data/cwkt/data/polyis-cache/noscope/$dataset/stride-15/" "$local_output/"
done
