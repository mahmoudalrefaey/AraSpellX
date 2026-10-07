import pytest

from araspellx.eval.categories import categorize
from araspellx.eval.metrics import regions_in, score, word_regions

SRC = "ذهبت الي الجامعه لاكن البيانت ناقصة"
REF = "ذهبت إلى الجامعة لكن البيانات ناقصة"


def test_regions_follow_merges_and_splits():
    source = "ومع الجتها المعلوماتالمطلوبة"
    target = "ومعالجتها المعلومات المطلوبة"
    regions = word_regions(source)
    assert [source[a:b] for a, b in regions] == ["ومع ", "الجتها ", "المعلوماتالمطلوبة"]
    assert regions_in(source, target, regions) == ["ومع", "الجتها ", "المعلومات المطلوبة"]


def test_perfect_output():
    result = score([SRC], [REF], [REF])
    assert (result.gold, result.system, result.correct) == (4, 4, 4)
    assert result.precision == result.recall == result.f05 == 1.0
    assert result.damage == 0.0
    assert result.rates()["cer_out"] == 0.0


def test_copying_the_input_scores_zero_recall_and_no_damage():
    result = score([SRC], [SRC], [REF])
    assert (result.gold, result.system, result.correct) == (4, 0, 0)
    assert result.recall == 0.0 and result.damage == 0.0


def test_wrong_fix_and_damage_are_counted():
    output = "ذهبت التي الجامعة لكن البيانات ناقص"  # الي->التي wrong, ناقصة damaged
    result = score([SRC], [output], [REF])
    assert (result.gold, result.system, result.correct) == (4, 5, 3)
    assert result.precision == pytest.approx(3 / 5)
    assert result.recall == pytest.approx(3 / 4)
    assert result.correct_words == 2 and result.damaged == 1
    assert result.damage == pytest.approx(0.5)


@pytest.mark.parametrize("source,target,expected", [
    ("اصبح", "أصبح", "hamza_alef"),
    ("السوال", "السؤال", "hamza_seat"),
    ("الجامعه", "الجامعة", "ta_marbuta"),
    ("علي", "على", "alef_maqsura"),
    ("يتمكنو", "يتمكنوا", "alef_fariqa"),
    ("نحسين", "تحسين", "dots"),
    ("المعلوماتالمطلوبة", "المعلومات المطلوبة", "spacing"),
    ("البيانت", "البيانات", "typo"),
    ("الاداره", "الإدارة", "mixed"),
])
def test_categories(source, target, expected):
    assert categorize(source, target) == expected
