# Tetris comparison setup

See [TETRIS.md](../TETRIS.md) for the command that runs both strides. The commands below prepare its inputs on `bombe`. The NoScope checkout starts at `6c6aa72e09280530dfdcf87871c1ac43df3b6cc3`. Check out [the pinned TensorFlow inference source](https://github.com/stanford-futuredata/tensorflow-noscope/tree/1b4a96364a79924f5c127bd21f8379b909d86f46) as `speedhax/` in the NoScope checkout.

## Prepare the inputs and CNNs

Stage each dataset under `/data/cwkt/data/noscope-pilot/<dataset>/` with the split clips in `videos/{train,valid,test}/`, the cached Tetris reference detections in `reference/<clip>/002_naive/detection.jsonl`, and the Seiden timing manifest in `seiden_manifest.json`. From `tetris_comparison/` on `bombe`, set `dataset` to `caldot1-y05`, `caldot2-y05`, `ams-y05`, `jnc0`, `jnc2`, `jnc6`, or `jnc7`.

```sh
./run.sh prepare_legacy.py "/data/$dataset" "/data/$dataset/released-inputs" train valid test
docker build -f Dockerfile.legacy-train -t noscope-legacy-train .
docker build -f Dockerfile.compat-train-gpu -t noscope-compat-train-gpu:official .
NOSCOPE_DATASET="$dataset" ./train_all_gpus.sh 0 1 2 3 4 6 7
```

Pass the idle GPUs available on the host. For concurrent dataset training, pass disjoint GPU lists to the separate `train_all_gpus.sh` invocations. The wrapper fixes the training split seed across workers and distributes the released 12-candidate grid. It uses Keras 1.2.2 with TensorFlow 1.15 and CUDA 10 because legacy cuDNN failed during backpropagation on the RTX 6000. After the workers finish, export the H5 files with the released TensorFlow 0.12 exporter:

```sh
shards="/data/cwkt/data/noscope-pilot/$dataset/compat-gpu-shards"
docker run --rm -v "$(cd .. && pwd):/noscope:ro" -v "$shards:/models" noscope-legacy-train \
    sh -ec 'for shard in /models/gpu-*; do
        python scripts/export-tf-graphs.py --model_dir "$shard" --output_dir "$shard"
    done'
```

## Build the released inference binary

```sh
docker build -f Dockerfile.released-inference -t noscope-released-inference .
./build_released.sh ../speedhax "/data/cwkt/data/noscope-pilot/$dataset/released-build"
mkdir -p "/data/cwkt/data/noscope-pilot/$dataset/released-cuda-cache"
```

The replay patch exposes the released engine's detector dry-run path and removes the Darknet link dependency. The wrapper fills requests from the cached detector and carries each decision forward. Validation uses complete 30-frame windows. For an incomplete test stride, it carries the last decision to the end so scoring covers the entire split.

Replay FPS includes measured frame differencing and CNN time plus the cached per-frame detector charge. Video loading and preparation are outside this timing boundary.
