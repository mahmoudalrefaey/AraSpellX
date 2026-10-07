"""The characters AraSpellX reads, and the subset it is allowed to edit.

Every character in VOCAB gets its own token; anything else is read as
[UNK] and is never edited. Only Arabic letters, tatweel, diacritics and
the space between words are editable (spec sections 2 and 4): digits,
Latin text, punctuation and symbols always pass through unchanged.
"""
from __future__ import annotations

import string
from typing import Dict, List

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
PAD, UNK, CLS, SEP, MASK = SPECIAL_TOKENS

# The 36 letters of standard Arabic orthography (U+0621-U+063A, U+0641-U+064A).
ARABIC_LETTERS = "".join(chr(c) for c in list(range(0x0621, 0x063B)) + list(range(0x0641, 0x064B)))

# Letters met in Arabic text outside standard orthography: alef wasla
# (classical and Quranic text) and letters of foreign names and loanwords
# (Persian/Urdu forms, Maghrebi/Gulf veh and qaf variants).
EXTRA_LETTERS = (
    "ٱ"  # ALEF WASLA
    "پ"  # PEH
    "چ"  # TCHEH
    "ژ"  # JEH
    "ڤ"  # VEH
    "ڨ"  # QAF WITH THREE DOTS ABOVE
    "ک"  # KEHEH
    "گ"  # GAF
    "ی"  # FARSI YEH
)

# Harakat, shadda, sukun, maddah/hamza marks and the superscript alef.
DIACRITICS = "".join(chr(c) for c in range(0x064B, 0x0656)) + chr(0x0670)

TATWEEL = chr(0x0640)

ARABIC_PUNCTUATION = (
    "،"  # ARABIC COMMA
    "؛"  # ARABIC SEMICOLON
    "؟"  # ARABIC QUESTION MARK
    "٪"  # ARABIC PERCENT SIGN
    "٫"  # ARABIC DECIMAL SEPARATOR
    "٬"  # ARABIC THOUSANDS SEPARATOR
    "۔"  # ARABIC FULL STOP
    "﴾"  # ORNATE LEFT PARENTHESIS
    "﴿"  # ORNATE RIGHT PARENTHESIS
)

DIGITS = (
    string.digits
    + "".join(chr(c) for c in range(0x0660, 0x066A))  # Arabic-Indic
    + "".join(chr(c) for c in range(0x06F0, 0x06FA))  # Extended Arabic-Indic (Persian)
)

LATIN_LETTERS = string.ascii_letters

SYMBOLS = string.punctuation + "«»“”‘’–—…•°×÷§¶©®™€£$¥"

WHITESPACE = " \n\t"

VOCAB: List[str] = SPECIAL_TOKENS + list(dict.fromkeys(
    ARABIC_LETTERS + EXTRA_LETTERS + DIACRITICS + TATWEEL + ARABIC_PUNCTUATION
    + DIGITS + LATIN_LETTERS + SYMBOLS + WHITESPACE
))

TOKEN_TO_ID: Dict[str, int] = {token: i for i, token in enumerate(VOCAB)}

ARABIC = frozenset(ARABIC_LETTERS + EXTRA_LETTERS)
EDITABLE = frozenset(ARABIC_LETTERS + EXTRA_LETTERS + DIACRITICS + TATWEEL + " ")


def is_arabic_letter(char: str) -> bool:
    return char in ARABIC


def is_editable(char: str) -> bool:
    """Whether the model may change, delete or insert next to this character."""
    return char in EDITABLE
