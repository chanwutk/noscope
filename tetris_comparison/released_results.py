"""Choose released NoScope plans on validation and collect their held-out replay outputs."""
import argparse
import csv
import json
import math
from pathlib import Path
from shutil import copyfile

import numpy as np

from replay_legacy import replay


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('plans', 'collect', 'retime'))
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
    assert set(by_target) == {0.25, 0.1, 0.05, 0.04, 0.03, 0.02, 0.01, 0.0}
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


def decision_trace(path):
    with path.open() as source:
        rows = csv.reader(line for line in source if not line.startswith('#'))
        return [(frame, status, label) for frame, status, _, _, _, label in rows if status != '1']


def retime(data, output):
    repeats = [output / f'repeat-{i}' for i in (1, 2, 3)]
    original = data / 'released-test'
    if not (original / 'run.json').is_file():
        original = repeats[0]
    result = json.loads((original / 'run.json').read_text())
    plans = json.loads((original / 'plans.json').read_text())
    runs = []
    with np.load(original / 'predictions.npz') as archive:
        selected = archive['selected']
    for repeat in repeats:
        assert json.loads((repeat / 'plans.json').read_text()) == plans, repeat
        with np.load(repeat / 'predictions.npz') as archive:
            assert np.array_equal(archive['selected'], selected), repeat
        run = json.loads((repeat / 'run.json').read_text())
        assert all(run[key] == result[key] for key in ('frames', 'clips', 'detector_ms_per_frame')), repeat
        assert len(run['records']) == len(plans), repeat
        runs.append(run['records'])

    original_device = (output / 'device.txt').read_text().strip()
    devices = {original_device}
    timing_keys = ('difference_seconds', 'cnn_seconds', 'fps')
    # Pick the measured repetition with the median sum of the two native timers.
    for i, (record, *trials) in enumerate(zip(result['records'], *runs, strict=True)):
        trace = decision_trace(original / f'test-{i}.csv')
        timings = []
        for repeat, trial in zip(repeats, trials, strict=True):
            assert decision_trace(repeat / f'test-{i}.csv') == trace, (repeat, i)
            assert all(trial[key] == record[key] for key in ('target', 'detector_frames', 'detector_seconds')), (repeat, i)
            assert math.isfinite(trial['fps']) and trial['fps'] > 0, (repeat, i)
            measured = {key: trial[key] for key in timing_keys}
            device_file = repeat / f'test-{i}.device'
            if device_file.is_file():
                device = device_file.read_text().strip()
                if device != original_device:
                    measured['device'] = device
                    devices.add(device)
            timings.append(measured)
        median = sorted(timings, key=lambda trial: trial['difference_seconds'] + trial['cnn_seconds'])[1]
        record.update({key: median[key] for key in timing_keys})
        record['timing_repeats'] = timings
    result['retiming'] = dict(device=original_device, executions='serial', statistic='median-of-three')
    if len(devices) > 1:
        result['retiming']['devices'] = sorted(devices)
    for name in ('plans.json', 'predictions.npz'):
        copyfile(original / name, output / name)
    temporary = output / 'run.json.tmp'
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(output / 'run.json')


def main():
    args = parse_args()
    plan_path = args.output / 'plans.json'
    if args.mode == 'plans':
        plans = choose_plans(args.data)
        if any(args.output.glob('test-*.csv')) and json.loads(plan_path.read_text()) != plans:
            raise ValueError('Selected plans changed. Use a fresh test output directory.')
        plan_path.write_text(json.dumps(plans, indent=2) + '\n')
    elif args.mode == 'collect':
        collect(args.data, args.output, json.loads(plan_path.read_text()))
    else:
        retime(args.data, args.output)


if __name__ == '__main__':
    main()
