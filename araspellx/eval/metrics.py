"""Edit-level scoring: precision, recall, F0.5, damage and error rates.

Scoring is done per source word. Each word owns a region of the source
text (the word plus the spaces after it); through a character alignment
we read what the reference and the model output made of that region:

- gold edit:    reference region != source region
- system edit:  output region != source region
- correct edit: output region == reference region != source region
- damage:       a correct source word (reference == source) that the
                output changed

Spaces belong to the word before them, so merging two words is an edit
of the first word and splitting a word is an edit of that word.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from rapidfuzz.distance import Levenshtein

from araspellx.eval.categories import categorize

WORD = re.compile(r"\S+")


def boundary_map(source: str, target: str) -> List[int]:
    """For every source position p (0..len), the target position before insertions at p."""
    mapping = [0] * (len(source) + 1)
    for op in Levenshtein.opcodes(source, target):
        if op.tag == "insert":
            continue
        for k in range(op.src_end - op.src_start):
            if op.tag == "delete":
                mapping[op.src_start + k] = op.dest_start
            else:
                mapping[op.src_start + k] = op.dest_start + k
    mapping[len(source)] = len(target)
    # Insertions at p belong to the region starting at p: map p to the target
    # position where those insertions start, i.e. right after the previous char.
    for op in Levenshtein.opcodes(source, target):
        if op.tag == "insert" and op.src_start < len(source):
            mapping[op.src_start] = op.dest_start
    return mapping


def word_regions(text: str) -> List[Tuple[int, int]]:
    """Each word's region: from its start to the start of the next word."""
    starts = [m.start() for m in WORD.finditer(text)]
    if not starts:
        return []
    starts[0] = 0
    return list(zip(starts, starts[1:] + [len(text)]))


def regions_in(source: str, target: str, regions: Sequence[Tuple[int, int]]) -> List[str]:
    mapping = boundary_map(source, target)
    return [target[mapping[a]:mapping[b]] for a, b in regions]


@dataclass
class Score:
    gold: int = 0
    system: int = 0
    correct: int = 0
    correct_words: int = 0   # source words that needed no edit
    damaged: int = 0         # ...of which the output changed
    char_errors_in: int = 0
    char_errors_out: int = 0
    word_errors_in: int = 0
    word_errors_out: int = 0
    ref_chars: int = 0
    ref_words: int = 0
    by_category: Dict[str, Counter] = field(default_factory=dict)

    @property
    def precision(self) -> float:
        return self.correct / self.system if self.system else 1.0

    @property
    def recall(self) -> float:
        return self.correct / self.gold if self.gold else 1.0

    @property
    def f05(self) -> float:
        p, r = self.precision, self.recall
        return 1.25 * p * r / (0.25 * p + r) if p + r else 0.0

    @property
    def damage(self) -> float:
        return self.damaged / self.correct_words if self.correct_words else 0.0

    def rates(self) -> Dict[str, float]:
        return {
            "cer_in": self.char_errors_in / max(self.ref_chars, 1),
            "cer_out": self.char_errors_out / max(self.ref_chars, 1),
            "wer_in": self.word_errors_in / max(self.ref_words, 1),
            "wer_out": self.word_errors_out / max(self.ref_words, 1),
        }

    def summary(self) -> Dict[str, float]:
        return {"precision": self.precision, "recall": self.recall, "f0.5": self.f05,
                "damage": self.damage, "gold_edits": self.gold, "system_edits": self.system,
                **self.rates()}


def score(sources: Sequence[str], outputs: Sequence[str], references: Sequence[str],
          category: Optional[Callable[[str, str], str]] = categorize) -> Score:
    result = Score()
    for source, output, reference in zip(sources, outputs, references):
        regions = word_regions(source)
        ref_regions = regions_in(source, reference, regions)
        out_regions = regions_in(source, output, regions)
        for (a, b), ref, out in zip(regions, ref_regions, out_regions):
            src = source[a:b]
            gold, system = ref != src, out != src
            result.gold += gold
            result.system += system
            correct = gold and out == ref
            result.correct += correct
            if not gold:
                result.correct_words += 1
                result.damaged += system
            if category is not None and (gold or system):
                name = category(src.strip(), (ref if gold else out).strip())
                counts = result.by_category.setdefault(name, Counter())
                counts["gold"] += gold
                counts["system"] += system
                counts["correct"] += correct
        result.char_errors_in += Levenshtein.distance(source, reference)
        result.char_errors_out += Levenshtein.distance(output, reference)
        result.word_errors_in += Levenshtein.distance(source.split(), reference.split())
        result.word_errors_out += Levenshtein.distance(output.split(), reference.split())
        result.ref_chars += len(reference)
        result.ref_words += len(reference.split())
    return result
