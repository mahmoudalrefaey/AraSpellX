"""Typed-error noise for real text (punctuation, digits and Latin included).

Two kinds of noise, both touching Arabic words only:

- habits: systematic spelling habits measured in real user text (dropped
  hamza on alef, ة/ه, ى/ي, hamza seats, ظ/ض, dropped alef after waw),
  each switched on per text and applied to most of its matches;
- typos: random keyboard-style errors inside words (insert, delete,
  swap, neighbouring key) and word spacing errors (merge, split), at a
  rate drawn per text.

Word ends are "not followed by an Arabic letter", so habits also apply
before punctuation (المدرسة، -> المدرسه،), unlike the v0 rules.
"""
from __future__ import annotations

import random
import re
from typing import Callable, List, Optional, Sequence, Tuple

from araspellx.text.charset import ARABIC_LETTERS

L = "ء-غف-ي"  # Arabic letters
END = f"(?![{L}])"
START = f"(?<![{L}])"
PREFIX = "(?:وال|بال|فال|كال|لل|ال|و|ف|ب|ل|ك)?"

# (name, pattern, replacement): regex habits, applied per match with a probability.
HABITS: List[Tuple[str, str, str]] = [
    ("hamza_alef", f"{START}({PREFIX})[أإآ]", r"\1ا"),
    ("ta_marbuta", f"ة{END}", "ه"),
    ("alef_maqsura", f"ى{END}", "ي"),
    ("hamza_waw", "ؤ", "و"),
    ("hamza_ya_after_alef", "(?<=ا)ئ", "ي"),
    ("hamza_after_alef", f"(?<=ا)ء(?=[{L}])", "ئ"),
    ("zah_dad", "ظ", "ض"),
    ("alef_fariqa", f"وا{END}", "و"),
]

KEYBOARD_ROWS = ["ضصثقفغعهخحجد", "شسيبلاتنمكط", "ئءؤرىةوزظ"]
ARABIC_WORD = re.compile(f"[{L}]+")


def _neighbours() -> dict:
    """Keys next to each letter on the Arabic keyboard (left, right, above, below)."""
    out = {}
    for r, row in enumerate(KEYBOARD_ROWS):
        for c, char in enumerate(row):
            near = set(row[max(c - 1, 0):c + 2]) - {char}
            for rr in (r - 1, r + 1):
                if 0 <= rr < len(KEYBOARD_ROWS):
                    near |= set(KEYBOARD_ROWS[rr][max(c - 1, 0):c + 2])
            out[char] = sorted(near - {char})
    return out


NEIGHBOURS = _neighbours()


def apply_habits(text: str, rng: random.Random, habit_prob: float = 0.5,
                 apply_prob: float = 0.9, only: Optional[Sequence[str]] = None) -> str:
    for name, pattern, replacement in HABITS:
        if (only is not None and name not in only) or (only is None and rng.random() >= habit_prob):
            continue
        text = re.sub(pattern, lambda m: m.expand(replacement) if rng.random() < apply_prob else m.group(),
                      text)
    return text


def _typo(text: str, rng: random.Random) -> str:
    words = [m.span() for m in ARABIC_WORD.finditer(text)]
    if not words:
        return text
    start, end = rng.choice(words)
    i = rng.randrange(start, end)
    kind = rng.choice(["insert", "delete", "swap", "keyboard", "merge", "split"])
    if kind == "insert":
        return text[:i] + rng.choice(ARABIC_LETTERS) + text[i:]
    if kind == "delete" and end - start > 2:
        return text[:i] + text[i + 1:]
    if kind == "swap" and i + 1 < end and text[i] != text[i + 1]:
        return text[:i] + text[i + 1] + text[i] + text[i + 2:]
    if kind == "keyboard" and NEIGHBOURS.get(text[i]):
        return text[:i] + rng.choice(NEIGHBOURS[text[i]]) + text[i + 1:]
    if kind == "merge" and end < len(text) - 1 and text[end] == " " and text[end + 1] in ARABIC_LETTERS:
        return text[:end] + text[end + 1:]
    if kind == "split" and start + 1 < i < end - 1:
        return text[:i] + " " + text[i:]
    return text


def add_typos(text: str, rng: random.Random, ratio: float) -> str:
    """Roughly `ratio` typo operations per Arabic letter."""
    letters = sum(c in ARABIC_LETTERS for c in text)
    for _ in range(int(ratio * letters)):
        text = _typo(text, rng)
    return text


def typed_noise(text: str, rng: random.Random,
                typo_ratios: Sequence[float] = (0.0, 0.1, 0.15, 0.15)) -> str:
    """Habits plus typos at a per-text ratio (the v0 generator's tuned defaults)."""
    return add_typos(apply_habits(text, rng), rng, rng.choice(typo_ratios))


NoiseFn = Callable[[str, random.Random], str]
