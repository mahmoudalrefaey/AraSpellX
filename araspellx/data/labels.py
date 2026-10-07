"""Edit labels: one label per noisy character that turns it into clean text.

A label says what happens to its character and what is inserted right
after it:

    K       keep              D       delete
    R:x     replace with x    ...+s   then insert the string s after it

The [CLS] token in front of the text carries insertions before the first
character. Labels are derived by aligning the noisy and clean text
character by character. Only editable characters (Arabic letters,
diacritics, tatweel, space) may be changed or followed by insertions;
everything else is always kept, as the specification requires.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from rapidfuzz.distance import Levenshtein

from araspellx.text.charset import is_editable

KEEP, DELETE, REPLACE = "K", "D", "R"


def make_label(op: str, char: str = "", insert: str = "") -> str:
    label = f"{REPLACE}:{char}" if op == REPLACE else op
    return f"{label}+{insert}" if insert else label


def parse_label(label: str) -> Tuple[str, str, str]:
    """Split a label into (op, replacement char, inserted string)."""
    op = label[0]
    if op == REPLACE:
        char, rest = label[2], label[3:]
    else:
        char, rest = "", label[1:]
    insert = rest[1:] if rest.startswith("+") else ""
    return op, char, insert


@dataclass
class Labels:
    """Edit labels of one noisy text: `cls` for the [CLS] token, `chars[i]` for noisy[i]."""
    cls: str
    chars: List[str]


def derive_labels(noisy: str, clean: str, max_insert: int = 2) -> Labels:
    """Labels that turn noisy into clean (insertions longer than max_insert are cut)."""
    ops = [KEEP] * len(noisy)
    replacement = [""] * len(noisy)
    inserted_before = [""] * (len(noisy) + 1)  # [i]: text inserted before noisy[i]
    for op in Levenshtein.opcodes(noisy, clean):
        if op.tag == "replace":
            for k in range(op.src_end - op.src_start):
                ops[op.src_start + k] = REPLACE
                replacement[op.src_start + k] = clean[op.dest_start + k]
        elif op.tag == "delete":
            for i in range(op.src_start, op.src_end):
                ops[i] = DELETE
        elif op.tag == "insert":
            inserted_before[op.src_start] += clean[op.dest_start:op.dest_end]

    chars = []
    for i, char in enumerate(noisy):
        if not is_editable(char):
            chars.append(KEEP)
            continue
        insert = inserted_before[i + 1][:max_insert]
        chars.append(make_label(ops[i], replacement[i], insert))
    cls = make_label(KEEP, insert=inserted_before[0][:max_insert])
    return Labels(cls, chars)


def apply_labels(noisy: str, labels: Labels) -> str:
    """Apply labels to noisy text (the inverse of derive_labels when nothing was cut)."""
    out = [parse_label(labels.cls)[2]]
    for char, label in zip(noisy, labels.chars):
        op, replacement, insert = parse_label(label)
        if op == KEEP:
            out.append(char)
        elif op == REPLACE:
            out.append(replacement)
        out.append(insert)
    return "".join(out)


class LabelVocab:
    """The fixed set of labels the classifier predicts (Hugging Face id2label)."""

    def __init__(self, labels: List[str]) -> None:
        if labels[0] != KEEP:
            raise ValueError("KEEP must be label 0")
        self.labels = labels
        self.index: Dict[str, int] = {label: i for i, label in enumerate(labels)}

    @classmethod
    def build(cls, counts: Counter, coverage: float = 0.995, min_count: int = 20) -> "LabelVocab":
        """Keep the most frequent edit labels covering `coverage` of all edit occurrences."""
        edits = [(label, n) for label, n in counts.most_common() if label != KEEP and n >= min_count]
        total = sum(n for label, n in counts.items() if label != KEEP)
        kept, covered = [], 0
        for label, n in edits:
            if total and covered / total >= coverage:
                break
            kept.append(label)
            covered += n
        return cls([KEEP] + kept)

    def encode(self, label: str) -> int:
        """Id of a label; unknown labels fall back to dropping their insertion, then to KEEP."""
        if label in self.index:
            return self.index[label]
        op, char, _ = parse_label(label)
        return self.index.get(make_label(op, char), 0)

    def decode(self, label_id: int) -> str:
        return self.labels[label_id]

    def __len__(self) -> int:
        return len(self.labels)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.labels, ensure_ascii=False, indent=0), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "LabelVocab":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))


def count_labels(pairs: Iterable[Tuple[str, str]], max_insert: int = 2) -> Counter:
    counts: Counter = Counter()
    for noisy, clean in pairs:
        labels = derive_labels(noisy, clean, max_insert)
        counts[labels.cls] += 1
        counts.update(labels.chars)
    return counts
