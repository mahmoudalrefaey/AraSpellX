from collections import Counter

import pytest

from araspellx.data.labels import (
    KEEP, LabelVocab, apply_labels, derive_labels, make_label, parse_label)

PAIRS = [
    ("تطويير", "تطوير"),            # doubled letter
    ("البيانت", "البيانات"),          # missing letter
    ("الجامعه الي", "الجامعة إلى"),   # habits
    ("ومع الجتها", "ومعالجتها"),      # split word
    ("المعلوماتالمطلوبة", "المعلومات المطلوبة"),  # merged words
    ("يتمكنو", "يتمكنوا"),            # missing final alef
    ("لكتاب", "الكتاب"),              # insertion before the first character
    ("نحسين جميع", "تحسين جميع"),     # OCR dots
    ("سنة 2026 هي", "سنة 2026 هي"),   # nothing to do
]


@pytest.mark.parametrize("noisy,clean", PAIRS)
def test_labels_round_trip(noisy, clean):
    labels = derive_labels(noisy, clean)
    assert len(labels.chars) == len(noisy)
    assert apply_labels(noisy, labels) == clean


def test_label_examples():
    labels = derive_labels("البيانت", "البيانات")
    assert labels.chars[5] == "K+ا"  # insert alef after the nun
    assert derive_labels("لكتاب", "الكتاب").cls == "K+ا"
    assert derive_labels("الجامعه", "الجامعة").chars[6] == "R:ة"
    assert derive_labels("تطويير", "تطوير").chars[4] == "D"


def test_non_editable_characters_are_always_kept():
    labels = derive_labels("عام 2O26", "عام 2026")  # Latin O where a digit belongs
    assert labels.chars[5] == KEEP


def test_long_insertions_are_cut_for_later_passes():
    labels = derive_labels("ب", "بسمل", max_insert=2)
    assert labels.chars[0] == "K+سم"


def test_parse_and_make_are_inverse():
    for label in ["K", "D", "R:ة", "K+ا", "R:ي+ ", "D+ال"]:
        op, char, insert = parse_label(label)
        assert make_label(op, char, insert) == label


def test_vocab_coverage_and_fallbacks():
    counts = Counter({"K": 1000, "R:ة": 50, "K+ا": 30, "D": 25, "R:ة+ ": 1})
    vocab = LabelVocab.build(counts, coverage=0.99, min_count=10)
    assert vocab.labels == ["K", "R:ة", "K+ا", "D"]
    assert vocab.encode("R:ة+ ") == vocab.index["R:ة"]  # rare: insertion dropped
    assert vocab.encode("R:ڤ") == 0                    # unknown: keep
