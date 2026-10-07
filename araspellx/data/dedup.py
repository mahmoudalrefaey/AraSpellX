"""Keep test-set text out of training data with word shingles.

Every test set's text is cut into overlapping 8-word shingles (after light
normalization: no diacritics, tatweel or punctuation). A training paragraph
that shares any shingle with a test set is dropped. This catches the same
sentence even when it appears in a different revision, page or source
(several test sets are Wikipedia text, and OpenITI books may also be on
Wikisource).
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Set

from araspellx.text.charset import ARABIC, DIACRITICS, TATWEEL

SHINGLE = 8
NOT_LETTERS = re.compile(r"[^\w\s]")


def _words(text: str) -> list:
    text = "".join(c for c in text if c not in DIACRITICS and c != TATWEEL)
    return NOT_LETTERS.sub(" ", text).split()


def shingles(text: str) -> Iterator[int]:
    words = _words(text)
    for i in range(max(len(words) - SHINGLE + 1, 0)):
        chunk = " ".join(words[i:i + SHINGLE])
        if sum(c in ARABIC for c in chunk) >= 20:
            yield int.from_bytes(hashlib.blake2b(chunk.encode(), digest_size=8).digest(), "big")


def test_shingles(paths: Iterable[Path], fields=("text", "clean")) -> Set[int]:
    """Shingles of every text field of every JSONL test file (or of whole .txt files)."""
    found: Set[int] = set()
    expanded = []
    for path in map(Path, paths):  # folders stand for every .txt/.jsonl file inside them
        expanded += sorted(p for p in path.rglob("*") if p.suffix in (".txt", ".jsonl")) if path.is_dir() else [path]
    for path in expanded:
        if Path(path).suffix == ".txt":
            found.update(shingles(Path(path).read_text(encoding="utf-8")))
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                for field in fields:
                    if field in row:
                        found.update(shingles(row[field]))
    return found


def overlaps(text: str, banned: Set[int]) -> bool:
    return any(s in banned for s in shingles(text))
