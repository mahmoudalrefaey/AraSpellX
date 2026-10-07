"""Name the kind of change between two versions of a word (for per-category scores).

Categories follow the specification: hamza on alef, hamza seat, ta marbuta,
alef maqsura, alef after waw, dots (same letter shape, different dots,
typical of OCR), spacing (words merged or split), and other typos.
"""
from __future__ import annotations

from rapidfuzz.distance import Levenshtein

ALEFS = set("اأإآٱ")
SEATS = set("ؤئءوي")
# Letters that share a shape and differ only in dots (rasm groups).
SHAPES = ["بتثنيىئ", "جحخ", "دذ", "رز", "سش", "صض", "طظ", "عغ", "فق", "هة"]
SHAPE_OF = {letter: i for i, group in enumerate(SHAPES) for letter in group}


def _substitution(a: str, b: str, end: bool) -> str:
    if a in ALEFS and b in ALEFS:
        return "hamza_alef"
    if {a, b} == {"ة", "ه"} and end:
        return "ta_marbuta"
    if {a, b} == {"ى", "ي"} and end:
        return "alef_maqsura"
    if a in SEATS and b in SEATS:
        return "hamza_seat"
    if a in SHAPE_OF and SHAPE_OF.get(b) == SHAPE_OF[a]:
        return "dots"
    return "typo"


def categorize(source: str, target: str) -> str:
    """Category of the change source -> target (both one word region, stripped)."""
    if source.replace(" ", "") == target.replace(" ", ""):
        return "spacing"
    names = set()
    for op in Levenshtein.editops(source, target):
        if op.tag == "replace":
            names.add(_substitution(source[op.src_pos], target[op.dest_pos],
                                    end=op.src_pos == len(source) - 1))
        elif op.tag == "insert" and target[op.dest_pos] == " ":
            names.add("spacing")
        elif op.tag == "delete" and source[op.src_pos] == " ":
            names.add("spacing")
        elif op.tag == "insert" and target.endswith("وا") and target[op.dest_pos] == "ا" \
                and op.dest_pos == len(target) - 1:
            names.add("alef_fariqa")
        else:
            names.add("typo")
    if len(names) == 1:
        return names.pop()
    return "mixed"
