# NoScope for the Tetris comparison

From the Tetris workspace root, pass the index of an idle GPU on `bombe`.

```sh
./noscope/tetris_comparison/run_comparison.sh 0
```

The command runs CalDoT1, CalDoT2, Amsterdam, and B3D1–B3D4 (`jnc0`, `jnc2`, `jnc6`, `jnc7`). To run a subset, pass the dataset names after the GPU index.

```sh
./noscope/tetris_comparison/run_comparison.sh 0 caldot1-y05
```

Stage the inputs, trained CNNs, and inference binary using the [setup guide](tetris_comparison/README.md). The runner reuses completed stages and scores predictions with the §7.3 evaluator in the `polyis` container on `ace`.

For each dataset, it profiles the same 12 CNNs at strides **5 and 15** using 30-frame validation windows. It tunes the released global-MSE cascade separately for each stride at the default paired FP/FN targets of 0%, 1%, 2%, 3%, 4%, 5%, 10%, and 25%.

Each stride and target runs three times on the specified GPU, one trial at a time. The runner verifies identical decisions across repeats and reports the median combined frame-difference and CNN time plus the cached detector charge. Each replacement trial records its device if resumed on a different GPU of the same model.

Results go under `tetris/measurements-results/r7o3_noscope/<dataset>/`.

- `stride-5-serial/scores.json` and `stride-15-serial/scores.json` contain every held-out score and timing trial.
- `scores-selected.json` contains the held-out score of the plan with the lowest validation `optimizer_cost` across both strides for each target. Equal costs favor the shorter stride.

For the paper comparison, use the FPS–F1 Pareto frontier of `scores-selected.json` and pick the point closest to LEAP in FPS. Break ties by higher F1, then lower validation error target.
