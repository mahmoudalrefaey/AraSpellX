"""Training windows for the correction model: noisy characters plus one label each.

Each window is drawn from one of these sources (shares from the design,
renormalized over the sources that exist):

- clean:   unchanged training paragraphs (teaches "leave correct text alone";
           includes dialect and diacritized text)
- typed:   the same paragraphs with typed-error noise applied on the fly
- ocr:     (OCR output, clean) pairs: rendered pages and real Yarmouk scans
- edits:   real corrections mined from Wikipedia history (when available);
           only the corrected words are learned from, because an edit leaves
           the paragraph's other mistakes in place

Clean paragraphs come from the pretraining stream, which is already
normalized, deduplicated and filtered against every test set. Pair files
are normalized here the same way inference normalizes its input.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from araspellx.data.labels import KEEP, LabelVocab, derive_labels, count_labels
from araspellx.noise.typed import typed_noise
from araspellx.text.charset import CLS, SEP, TOKEN_TO_ID, VOCAB
from araspellx.text.normalize import normalize
from araspellx.text.tokenizer import encode_chars

WINDOW = 512
CONTENT = WINDOW - 2
SHARES = {"clean": 0.35, "typed": 0.25, "ocr": 0.30, "edits": 0.10}
NEWLINE = TOKEN_TO_ID["\n"]


class Paragraphs:
    """Random clean paragraphs from a one-byte-per-character token stream."""

    def __init__(self, path: Path) -> None:
        self.data = np.memmap(path, dtype=np.uint8, mode="r")

    def sample(self, rng: random.Random, min_chars: int = 40) -> str:
        while True:
            start = rng.randrange(len(self.data) - 2 * CONTENT)
            chunk = np.asarray(self.data[start:start + 2 * CONTENT])
            breaks = np.flatnonzero(chunk == NEWLINE)
            if len(breaks) >= 2 and breaks[1] - breaks[0] > min_chars:
                ids = chunk[breaks[0] + 1:breaks[1]]
                return "".join(VOCAB[i] for i in ids)


def read_rows(paths: Sequence[Path]) -> List[Dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    return rows


def normalized_pairs(rows: Sequence[Dict]) -> List[Tuple[str, str]]:
    """(noisy, clean) pairs normalized the way inference normalizes its input."""
    return [(normalize(r["noisy"]).text, normalize(r["clean"]).text) for r in rows]


def load_pairs(paths: Sequence[Path]) -> List[Tuple[str, str]]:
    return normalized_pairs(read_rows(paths))


class Mixture:
    """Draws (noisy text, clean text, source) examples in the configured shares."""

    def __init__(self, paragraphs: Paragraphs, ocr: List[Tuple[str, str]],
                 edits: Optional[List[Tuple[str, str]]] = None) -> None:
        self.paragraphs, self.ocr, self.edits = paragraphs, ocr, edits or []
        shares = {k: v for k, v in SHARES.items() if k in ("clean", "typed") or getattr(self, k)}
        total = sum(shares.values())
        self.sources = list(shares)
        self.weights = [shares[k] / total for k in self.sources]

    def draw(self, rng: random.Random) -> Tuple[str, str, str]:
        source = rng.choices(self.sources, self.weights)[0]
        if source in ("ocr", "edits"):
            noisy, clean = rng.choice(getattr(self, source))
            return noisy, clean, source
        clean = self.paragraphs.sample(rng)
        return (typed_noise(clean, rng) if source == "typed" else clean), clean, source


def fixed_words(text: str, char_labels: Sequence[str]) -> List[bool]:
    """Characters of the words that a label changes (both neighbours of a changed space)."""
    covered = [False] * len(text)
    for i, label in enumerate(char_labels):
        if label == KEEP:
            continue
        start, end = i, i + 1  # grow to the word around i (for a space: the words on both sides)
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        while end < len(text) and not text[end].isspace():
            end += 1
        covered[start:end] = [True] * (end - start)
    return covered


def window_example(noisy: str, clean: str, vocab: LabelVocab, rng: random.Random, partial: bool = False):
    """Input ids and label ids of one window ([CLS] + up to 510 characters + [SEP]).

    partial: `clean` fixes only some of the errors of `noisy` (a real edit leaves the
    paragraph's other mistakes in place), so only the changed words are learned from;
    every other character is ignored by the loss instead of being taught as correct.
    """
    labels = derive_labels(noisy, clean)
    covered = fixed_words(noisy, labels.chars) if partial else None
    if len(noisy) <= CONTENT:
        start = 0
    elif partial and any(covered):  # make the window include a fixed word
        anchor = rng.choice([i for i, c in enumerate(covered) if c])
        start = min(max(anchor - rng.randrange(CONTENT), 0), len(noisy) - CONTENT)
    else:
        start = rng.randrange(len(noisy) - CONTENT + 1)
    chars = noisy[start:start + CONTENT]
    char_labels = labels.chars[start:start + CONTENT]
    cls_label = labels.cls if start == 0 else KEEP
    ids = [TOKEN_TO_ID[CLS]] + encode_chars(chars) + [TOKEN_TO_ID[SEP]]
    label_ids = [vocab.encode(l) for l in char_labels]
    if partial:
        label_ids = [l if c else -100 for l, c in zip(label_ids, covered[start:start + CONTENT])]
        cls_id = vocab.encode(cls_label) if cls_label != KEEP else -100
    else:
        cls_id = vocab.encode(cls_label)
    return ids, [cls_id] + label_ids + [-100]


def build_vocab(mixture: Mixture, samples: int, seed: int = 0) -> LabelVocab:
    rng = random.Random(seed)
    pairs = []
    for _ in range(samples):
        noisy, clean, _ = mixture.draw(rng)
        pairs.append((noisy, clean))
    return LabelVocab.build(count_labels(pairs))


def batch(mixture: Mixture, vocab: LabelVocab, size: int, rng: random.Random):
    import torch

    rows = []
    for _ in range(size):
        noisy, clean, source = mixture.draw(rng)
        rows.append(window_example(noisy, clean, vocab, rng, partial=source == "edits"))
    length = max(len(ids) for ids, _ in rows)
    inputs = torch.zeros(size, length, dtype=torch.long)
    labels = torch.full((size, length), -100, dtype=torch.long)
    for i, (ids, label_ids) in enumerate(rows):
        inputs[i, :len(ids)] = torch.tensor(ids)
        labels[i, :len(label_ids)] = torch.tensor(label_ids)
    return inputs, labels
