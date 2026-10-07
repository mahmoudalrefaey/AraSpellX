"""Build the pretraining token stream from the extracted Wikimedia corpora.

Paragraphs from training pages only (see data/splits.py) are normalized
like inference input, deduplicated, filtered against every test set (word
shingles, data/dedup.py) and written as one stream of character ids (one
byte each; paragraphs separated by a newline) that training reads directly
from disk. One paragraph in 200 goes to a validation stream instead.

    python -m araspellx.data.pretrain_corpus --tests data/v1/testsets/*.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Iterator, List, Set, Tuple

import numpy as np

from araspellx.data.dedup import overlaps, test_shingles
from araspellx.data.splits import is_training
from araspellx.text.charset import VOCAB
from araspellx.text.normalize import normalize
from araspellx.text.tokenizer import encode_chars

assert len(VOCAB) < 256, "ids are stored as single bytes"

# Source corpora and the most characters taken from each. Egyptian Wikipedia
# is mostly bot-generated stubs, so it is capped.
SOURCES = {
    "arwiki": None,
    "arwikisource": None,
    "arywiki": None,
    "arzwiki": 150_000_000,
}
NEWLINE = encode_chars("\n")[0]


_banned: Set[int] = set()


def _init(banned: Set[int]) -> None:
    global _banned
    _banned = banned


def _process_page(job: Tuple[str, str]) -> List[Tuple[bytes, bytes, int, bool]]:
    """(dedup key, token bytes, characters, overlaps a test set) for each paragraph of a training page."""
    wiki, line = job
    page = json.loads(line)
    if not is_training(wiki, page["page_id"]):
        return []
    result = []
    for paragraph in page["paragraphs"]:
        text = normalize(paragraph).text
        key = hashlib.blake2b(text.encode(), digest_size=8).digest()
        ids = np.array(encode_chars(text) + [NEWLINE], dtype=np.uint8)
        result.append((key, ids.tobytes(), len(text), overlaps(text, _banned)))
    return result


def _lines(path: Path, wiki: str) -> Iterator[Tuple[str, str]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield wiki, line


def build(corpus: Path, tests: List[Path], out: Path, workers: int = 10) -> None:
    out.mkdir(parents=True, exist_ok=True)
    banned = test_shingles(tests)
    print(f"{len(banned)} test shingles from {len(tests)} test paths", file=sys.stderr)
    seen = set()
    stats = {}
    with open(out / "train.bin", "wb") as train, open(out / "valid.bin", "wb") as valid, \
            Pool(workers, initializer=_init, initargs=(banned,)) as pool:
        for wiki, cap in SOURCES.items():
            kept_chars = dropped_test = dropped_dup = 0
            for paragraphs in pool.imap(_process_page, _lines(corpus / f"{wiki}.jsonl", wiki), chunksize=64):
                if cap is not None and kept_chars >= cap:
                    break
                for key, data, chars, in_test in paragraphs:
                    if key in seen:
                        dropped_dup += 1
                        continue
                    seen.add(key)
                    if in_test:
                        dropped_test += 1
                        continue
                    (valid if key[0] < 2 else train).write(data)  # ~1/128 to validation
                    kept_chars += chars
            stats[wiki] = {"chars": kept_chars, "dropped_test_overlap": dropped_test,
                           "dropped_duplicates": dropped_dup}
            print(f"{wiki}: {stats[wiki]}", file=sys.stderr, flush=True)
    stats["train_bytes"] = (out / "train.bin").stat().st_size
    stats["valid_bytes"] = (out / "valid.bin").stat().st_size
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/v1/corpus"))
    parser.add_argument("--tests", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("data/v1/pretrain"))
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    build(args.corpus, args.tests, args.out, args.workers)


if __name__ == "__main__":
    main()
