"""Choose released NoScope plans on validation and collect their held-out replay outputs."""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from replay_legacy import replay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('plans', 'collect'))
    parser.add_argument('data', type=Path)
    parser.add_argument('output', type=Path)
    return parser.parse_args()


def choose_plans(data):
    # Keep NoScope's cost-based choice for each validation target.
    by_target = {}
    for profile in sorted((data / 'released-calibration').glob('convnet_*.json')):
        model, = data.glob(f'compat-gpu-shards/gpu-*/*_{profile.stem}.pb')
        for plan in json.loads(profile.read_text()):
            target = plan['target']
            cost = plan['optimizer_cost']
            if math.isfinite(cost) and (target not in by_target or cost < by_target[target]['optimizer_cost']):
                by_target[target] = dict(plan, graph=str(model.relative_to(data)),
                                         stride=plan['threshold_skip_distance'])
    assert set(by_target) == {0.0, 0.01, 0.02}
    return [by_target[target] for target in sorted(by_target)]


def collect(data, output, plans):
    # Verify the layout against the same detector-timing manifest used by Seiden.
    manifest = json.loads((data / 'seiden_manifest.json').read_text())
    frames = int((data / 'released-inputs' / 'test.frames').read_text())
    assert manifest['frames'] == frames
    with (data / 'released-inputs' / 'test.csv').open() as source:
        reference = {int(row['frame']) for row in csv.DictReader(source) if row['object_name'] == 'car'}
    records, predictions = [], []
    for index, plan in enumerate(plans):
        held_frames = frames % plan['stride']
        with (output / f'test-{index}.csv').open() as source:
            header = next(source)
            diff, cnn, _, _ = map(float, header.split('runtime:', 1)[1].split())
            decision = replay(csv.reader(line for line in source if not line.startswith('#')),
                              reference, frames - held_frames)
        selected = decision['selected']
        # Carry the last decision through any incomplete stride to score every test frame.
        if held_frames:
            selected.extend([selected[-1]] * held_frames)
        fallback = decision['detector_frames']
        detect = fallback * manifest['detector_ms_per_frame'] / 1000
        records.append(dict(plan, detector_frames=fallback, tail_held_frames=held_frames,
                            difference_seconds=diff, cnn_seconds=cnn,
                            detector_seconds=detect, fps=frames / (diff + cnn + detect)))
        predictions.append(selected)
    np.savez_compressed(output / 'predictions.npz', selected=predictions)
    result = dict(frames=frames, clips=[(c['video'], c['frames']) for c in manifest['clips']],
                  detector_ms_per_frame=manifest['detector_ms_per_frame'], records=records)
    (output / 'run.json').write_text(json.dumps(result, indent=2) + '\n')


def main():
    args = parse_args()
    plan_path = args.output / 'plans.json'
    if args.mode == 'plans':
        plans = choose_plans(args.data)
        if any(args.output.glob('test-*.csv')) and json.loads(plan_path.read_text()) != plans:
            raise ValueError('Selected plans changed. Use a fresh test output directory.')
        plan_path.write_text(json.dumps(plans, indent=2) + '\n')
    else:
        collect(args.data, args.output, json.loads(plan_path.read_text()))


if __name__ == '__main__':
    main()
