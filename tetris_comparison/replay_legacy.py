"""Fill only the detector requests emitted by the released NoScope replay mode."""
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('confidences', type=Path)
    parser.add_argument('reference', type=Path)
    parser.add_argument('frames', type=int)
    parser.add_argument('output', type=Path)
    return parser.parse_args()


def replay(rows, reference, frames):
    # speedhax emits one row per source frame. Skip rows repeat the preceding sampled decision.
    selected = []
    calls = 0
    for index, row in enumerate(rows):
        frame, status, _, _, _, label = row
        status = int(status)
        expected_frame = index if status == 1 else index + 1
        if int(frame) != expected_frame or status not in (1, 2, 4, 6):
            raise ValueError(f'Unexpected speedhax output at frame {index}: {row}')
        if status == 1:
            if index == 0:
                raise ValueError('First frame is skipped')
            selected.append(selected[-1])
        elif status == 6:
            selected.append(index in reference)
            calls += 1
        else:
            selected.append(bool(int(label)))
    if len(selected) != frames:
        raise ValueError(f'Expected {frames} output frames, got {len(selected)}')
    return dict(selected=selected, detector_frames=calls)


def main():
    args = parse_args()
    # Keep the cached reference labels outside NoScope's filter and CNN execution.
    with args.reference.open() as source:
        reference = {int(row['frame']) for row in csv.DictReader(source) if row['object_name'] == 'car'}
    with args.confidences.open() as source:
        rows = csv.reader(line for line in source if not line.startswith('#'))
        result = replay(rows, reference, args.frames)
    args.output.write_text(json.dumps(result) + '\n')


if __name__ == '__main__':
    main()
