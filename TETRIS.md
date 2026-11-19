# NoScope for the Tetris comparison

Run this command from the Tetris workspace root.

```sh
./noscope/tetris_comparison/run_stride15.sh
```

This runs CalDoT1, CalDoT2, and Amsterdam in order. To run just CalDoT1, pass its dataset name.

```sh
./noscope/tetris_comparison/run_stride15.sh caldot1-y05
```

The command expects the released inference binary, the 12 trained CNN graphs, the prepared validation and test inputs, and the detector timing manifest staged on `bombe`. It uses the §7.3 scorer in the `polyis` container on `ace`. See [environment and training setup](tetris_comparison/README.md) to prepare these inputs.

For each dataset, the command profiles all 12 CNNs at stride 15 on validation, tunes the released global-MSE cascade at the paired 0%, 1%, and 2% FP/FN targets, then scores the selected configurations on held-out frames. It reuses completed stages and does not retrain the CNNs. Outputs are under `tetris/measurements-results/r7o3_noscope/<dataset>/stride-15/`, including `scores.json`.
