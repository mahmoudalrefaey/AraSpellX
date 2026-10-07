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
from pathlib import Path
from typing import List

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


def build(corpus: Path, tests: List[Path], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    banned = test_shingles(tests)
    print(f"{len(banned)} test shingles from {len(tests)} test files", file=sys.stderr)
    seen = set()
    stats = {}
    with open(out / "train.bin", "wb") as train, open(out / "valid.bin", "wb") as valid:
        for wiki, cap in SOURCES.items():
            kept_chars = dropped_test = dropped_dup = 0
            with open(corpus / f"{wiki}.jsonl", encoding="utf-8") as f:
                for line in f:
                    if cap is not None and kept_chars >= cap:
                        break
                    page = json.loads(line)
                    if not is_training(wiki, page["page_id"]):
                        continue
                    for paragraph in page["paragraphs"]:
                        text = normalize(paragraph).text
                        key = hashlib.blake2b(text.encode(), digest_size=8).digest()
                        if key in seen:
                            dropped_dup += 1
                            continue
                        seen.add(key)
                        if overlaps(text, banned):
                            dropped_test += 1
                            continue
                        ids = np.array(encode_chars(text) + [NEWLINE], dtype=np.uint8)
                        (valid if key[0] < 2 else train).write(ids.tobytes())  # ~1/128 to validation
                        kept_chars += len(text)
            stats[wiki] = {"chars": kept_chars, "dropped_test_overlap": dropped_test,
                           "dropped_duplicates": dropped_dup}
            print(f"{wiki}: {stats[wiki]}", file=sys.stderr)
    stats["train_bytes"] = (out / "train.bin").stat().st_size
    stats["valid_bytes"] = (out / "valid.bin").stat().st_size
    (out / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/v1/corpus"))
    parser.add_argument("--tests", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, default=Path("data/v1/pretrain"))
    args = parser.parse_args()
    build(args.corpus, args.tests, args.out)


if __name__ == "__main__":
    main()
