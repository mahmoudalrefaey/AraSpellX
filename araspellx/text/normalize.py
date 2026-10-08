"""Deterministic normalization that keeps a map back to the original text.

Normalization folds characters that look like Arabic letters but are other
code points, expands presentation-form ligatures, composes letters written
as a base letter plus a combining hamza or madda (as some sources store
them), and removes tatweel and invisible control characters. Every change is
reported as an Edit in the
coordinates of the original text, and every normalized character remembers
the original span it came from, so edits made later on the normalized text
can be mapped back exactly (spec section 6).
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from araspellx.text.charset import DIACRITICS, TATWEEL

Span = Tuple[int, int]

# Look-alike code points folded into the standard Arabic letter.
LOOKALIKES = {
    chr(0x06CC): chr(0x064A),  # FARSI YEH -> YEH
    chr(0x06A9): chr(0x0643),  # KEHEH -> KAF
    chr(0x06C1): chr(0x0647),  # HEH GOAL -> HEH
    chr(0x06BE): chr(0x0647),  # HEH DOACHASHMEE -> HEH
    chr(0x06D5): chr(0x0647),  # AE -> HEH
    chr(0x06C3): chr(0x0629),  # TEH MARBUTA GOAL -> TEH MARBUTA
}

# A base letter followed by a combining hamza or madda, and the single letter it stands for.
COMPOSED = {
    (chr(0x0627), chr(0x0653)): chr(0x0622),  # ALEF + MADDAH ABOVE -> ALEF WITH MADDA ABOVE
    (chr(0x0627), chr(0x0654)): chr(0x0623),  # ALEF + HAMZA ABOVE -> ALEF WITH HAMZA ABOVE
    (chr(0x0627), chr(0x0655)): chr(0x0625),  # ALEF + HAMZA BELOW -> ALEF WITH HAMZA BELOW
    (chr(0x0648), chr(0x0654)): chr(0x0624),  # WAW + HAMZA ABOVE -> WAW WITH HAMZA ABOVE
    (chr(0x064A), chr(0x0654)): chr(0x0626),  # YEH + HAMZA ABOVE -> YEH WITH HAMZA ABOVE
    (chr(0x0649), chr(0x0654)): chr(0x0626),  # ALEF MAKSURA + HAMZA ABOVE -> YEH WITH HAMZA ABOVE
}

# Invisible characters: zero-width (non-)joiners and spaces, bidi controls, BOM.
INVISIBLE = frozenset(
    [chr(c) for c in range(0x200B, 0x2010)]
    + [chr(c) for c in range(0x202A, 0x202F)]
    + [chr(c) for c in range(0x2066, 0x206A)]
    + [chr(0xFEFF)]
)


def _is_presentation_form(char: str) -> bool:
    code = ord(char)
    return 0xFB50 <= code <= 0xFDFF or 0xFE70 <= code <= 0xFEFE


@dataclass
class Edit:
    """A change to the original text: replace text[start:end] with replacement."""
    start: int
    end: int
    original: str
    replacement: str
    confidence: float = 1.0
    category: str = "normalization"


@dataclass
class Normalized:
    """Normalized text plus the original span of every normalized character."""
    text: str
    spans: List[Span]
    original_length: int
    edits: List[Edit] = field(default_factory=list)

    def to_original(self, start: int, end: int) -> Span:
        """Map the span [start, end) of the normalized text to the original text.

        An empty span is an insertion point; it maps to the original position
        just before the normalized character at `start`.
        """
        if start < end:
            return self.spans[start][0], self.spans[end - 1][1]
        if start < len(self.spans):
            position = self.spans[start][0]
        elif self.spans:
            position = self.spans[-1][1]
        else:
            position = self.original_length
        return position, position


def _normalize_char(char: str, remove_diacritics: bool) -> str:
    if char in INVISIBLE or char == TATWEEL:
        return ""
    if remove_diacritics and char in DIACRITICS:
        return ""
    if char in LOOKALIKES:
        return LOOKALIKES[char]
    if _is_presentation_form(char):
        return unicodedata.normalize("NFKC", char)
    return char


def _in_any(position: int, spans: Sequence[Span]) -> bool:
    return any(start <= position < end for start, end in spans)


def normalize(
        text: str,
        protected: Optional[Iterable[Span]] = None,
        remove_diacritics: bool = False,
        ) -> Normalized:
    """Normalize text; characters inside protected spans are left untouched."""
    protected = sorted(protected or [])
    chars: List[str] = []
    spans: List[Span] = []
    edits: List[Edit] = []
    run_start = None  # start of the current run of changed original characters
    run_replacement: List[str] = []

    def close_run(end: int) -> None:
        nonlocal run_start, run_replacement
        if run_start is not None:
            edits.append(Edit(run_start, end, text[run_start:end], "".join(run_replacement)))
            run_start, run_replacement = None, []

    i = 0
    while i < len(text):
        composed = COMPOSED.get(tuple(text[i:i + 2]))
        if composed and not _in_any(i, protected) and not _in_any(i + 1, protected):
            if run_start is None:
                run_start = i
            run_replacement.append(composed)
            chars.append(composed)
            spans.append((i, i + 2))
            i += 2
            continue
        char = text[i]
        replacement = char if _in_any(i, protected) else _normalize_char(char, remove_diacritics)
        if replacement == char:
            close_run(i)
        else:
            if run_start is None:
                run_start = i
            run_replacement.append(replacement)
        for new_char in replacement:
            chars.append(new_char)
            spans.append((i, i + 1))
        i += 1
    close_run(len(text))

    return Normalized("".join(chars), spans, len(text), edits)
