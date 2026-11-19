"""Export the staged Tetris splits as the released NoScope scripts' MP4 and CSV inputs."""
import argparse
import csv
import json
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('inputs', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('splits', nargs='+', choices=('train', 'valid', 'test'))
    return parser.parse_args()


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        # Keep Tetris clip order and detector frame identifiers.
        videos = sorted((args.inputs / 'videos' / split).glob('*.mp4'))
        if not videos:
            raise ValueError('No videos for ' + split)
        listing = args.output / (split + '.concat')
        listing.write_text(''.join(f"file '{path.resolve()}'\n" for path in videos))
        video = args.output / (split + '.mp4')
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                        '-i', str(listing), '-c', 'copy', str(video)], check=True)

        # The released optimizer treats any row for the target class as a positive.
        offset = 0
        with (args.output / (split + '.csv')).open('w', newline='') as output:
            writer = csv.writer(output)
            writer.writerow(('frame', 'object_name', 'confidence', 'xmin', 'ymin', 'xmax', 'ymax'))
            for clip in videos:
                labels = args.inputs / 'reference' / clip.name / '002_naive' / 'detection.jsonl'
                count = 0
                with labels.open() as source:
                    for count, line in enumerate(source, 1):
                        record = json.loads(line)
                        assert record['frame_idx'] == count - 1, labels
                        for x1, y1, x2, y2, confidence in record['detections']:
                            writer.writerow((offset + count - 1, 'car', confidence, x1, y1, x2, y2))
                offset += count

        # Verify the CSV offsets cover the matching concatenated video.
        frames = subprocess.check_output(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
             'stream=nb_frames', '-of', 'csv=p=0', str(video)], text=True
        ).strip()
        assert int(frames) == offset, f'{split}: video has {frames} frames, reference has {offset}'
        (args.output / (split + '.frames')).write_text(f'{offset}\n')
        print(split, offset, flush=True)


if __name__ == '__main__':
    main()
