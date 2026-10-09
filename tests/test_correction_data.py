"""Training windows from real edits learn only from the words the editor changed."""
import random

from araspellx.data.labels import KEEP, LabelVocab, count_labels
from araspellx.train.correction_data import fixed_words, window_example
from araspellx.data.labels import derive_labels

NOISY = "ذهب الطلاب الي الجامعه"
CLEAN = "ذهب الطلاب إلى الجامعه"  # only الي was fixed; الجامعه is still wrong


def _vocab(*pairs):
    return LabelVocab.build(count_labels(pairs), coverage=1.0, min_count=1)


def test_partial_windows_ignore_everything_but_the_fixed_words():
    vocab = _vocab((NOISY, CLEAN))
    _, labels = window_example(NOISY, CLEAN, vocab, random.Random(0), partial=True)
    chars = labels[1:-1]
    fixed = range(NOISY.index("الي"), NOISY.index("الي") + 3)
    assert all(chars[i] != -100 for i in fixed)
    assert any(chars[i] != vocab.encode(KEEP) for i in fixed)  # the fix itself is learned
    assert all(chars[i] == -100 for i in range(len(NOISY)) if i not in fixed)  # الجامعه is not taught as correct
    assert labels[0] == -100  # nothing is known about insertions at the start


def test_full_windows_teach_every_character():
    vocab = _vocab((NOISY, CLEAN))
    _, labels = window_example(NOISY, CLEAN, vocab, random.Random(0))
    assert -100 not in labels[:-1]


def test_a_merged_space_covers_both_words():
    noisy, clean = "ثم و يستخدم هنا", "ثم ويستخدم هنا"
    covered = fixed_words(noisy, derive_labels(noisy, clean).chars)
    assert "".join(c for c, keep in zip(noisy, covered) if keep) == "و يستخدم"
