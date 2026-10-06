"""Rebuild the distorted column of a dataset CSV with the real-world
distorter: spelling habits (hamza on alef, ta marbuta, alef maqsura,
hamza seat, zah/dad) followed by random typos at a per-sentence ratio.

The raw corpus is not part of the repository, so the distortions are
generated from the existing clean column. All original columns are kept
and the new column is appended, so the original distortions remain
available for regression comparisons.

Run from the repository root:

    python -m data.regenerate_distortions --input data/dataset/train.csv
    python -m data.regenerate_distortions --input data/dataset/test.csv
    python -m data.regenerate_distortions --input data/dataset/test.csv --preview 20
"""
from argparse import ArgumentParser
import csv
import random
import sys
from pathlib import Path
from data.processors import get_habits, get_real_world_distorter


def get_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        '--input', required=True, type=str,
        help='The dataset CSV to read the clean column from'
    )
    parser.add_argument(
        '--output', default=None, type=str,
        help='The CSV to write, defaults to <input>_rw.csv'
    )
    parser.add_argument(
        '--clean_key', default='clean', type=str,
        help='The csv column name of the clean items'
    )
    parser.add_argument(
        '--dist_key', default='distorted_rw', type=str,
        help='The csv column name to write the distorted items to'
    )
    parser.add_argument(
        '--habit_prob', default=0.5, type=float,
        help='Probability that a spelling habit is used in a sentence'
    )
    parser.add_argument(
        '--apply_prob', default=0.9, type=float,
        help='Probability that a used habit is applied to each of its matches'
    )
    parser.add_argument(
        '--typo_ratios', default=[0.0, 0.02, 0.05, 0.1], nargs='+', type=float,
        help='Typo ratios, one is drawn per sentence'
    )
    parser.add_argument(
        '--max_len', default=128, type=int,
        help='The --max_len used for training'
    )
    parser.add_argument(
        '--distortion_ratio', default=0.1, type=float,
        help='The --distortion_ratio used for training (sets the encoder length)'
    )
    parser.add_argument(
        '--seed', default=42, type=int
    )
    parser.add_argument(
        '--preview', default=0, type=int,
        help='Print this many samples per habit and combined, then exit'
    )
    parser.add_argument(
        '--preview_rows', default=20000, type=int,
        help='The number of rows scanned for the preview'
    )
    return parser


def get_max_distorted_chars(max_len: int, ratio: float) -> int:
    # ArabicData pads SOS + distorted text + EOS to
    # int(ratio * max_len) + max_len + 1 tokens.
    return int(ratio * max_len) + max_len - 1


def distort(distorter, clean: str, max_chars: int, retries: int = 10) -> str:
    for _ in range(retries):
        distorted = distorter.run(clean)
        if len(distorted) <= max_chars:
            return distorted
    # Habits never change the length of the sentence.
    return distorter.apply_habits(clean)


def read_clean(path: str, clean_key: str, n_rows: int) -> list:
    with open(path, encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return [row[clean_key] for _, row in zip(range(n_rows), reader)]


def preview(args, distorter) -> None:
    lines = read_clean(args.input, args.clean_key, args.preview_rows)
    random.shuffle(lines)
    for habit in get_habits(apply_prob=1.0):
        print(f'## {type(habit).__name__}: {habit.__doc__.split(":")[0].strip()}')
        shown = 0
        for line in lines:
            changed = habit.execute(line)
            if changed == line:
                continue
            print(f'clean    : {line}')
            print(f'distorted: {changed}')
            print()
            shown += 1
            if shown == args.preview:
                break
    print('## Combined: habits used with --habit_prob/--apply_prob, '
          'then typos at a ratio drawn from --typo_ratios')
    for line in lines[:args.preview]:
        print(f'clean    : {line}')
        print(f'distorted: {distort(distorter, line, args.max_chars)}')
        print()


def main(args) -> None:
    random.seed(args.seed)
    args.max_chars = get_max_distorted_chars(args.max_len, args.distortion_ratio)
    distorter = get_real_world_distorter(
        habit_prob=args.habit_prob,
        apply_prob=args.apply_prob,
        typo_ratios=args.typo_ratios
    )
    if args.preview > 0:
        preview(args, distorter)
        return

    output = args.output or str(
        Path(args.input).with_name(Path(args.input).stem + '_rw.csv')
    )
    csv.field_size_limit(10 ** 8)
    with open(args.input, encoding='utf-8', newline='') as fin, \
            open(output, 'w', encoding='utf-8', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        if args.dist_key in header:
            raise ValueError(f'{args.input} already has a {args.dist_key} column')
        clean_idx = header.index(args.clean_key)
        writer.writerow(header + [args.dist_key])
        for i, row in enumerate(reader, 1):
            row.append(distort(distorter, row[clean_idx], args.max_chars))
            writer.writerow(row)
            if i % 500000 == 0:
                print(f'{i} rows', file=sys.stderr)
    print(f'Saved {output}')


if __name__ == '__main__':
    parser = get_argparser()
    args = parser.parse_args()
    main(args)
