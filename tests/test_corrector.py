"""Corrections are reported per word, with spans in the caller's text."""
from araspellx.correct.corrector import build_result, calibrated
from araspellx.correct.decode import Prediction
from araspellx.data.labels import DELETE, KEEP, REPLACE, make_label
from araspellx.text.normalize import normalize

TATWEEL = chr(0x0640)


def _result(text, edits, threshold=0.5, calibration=None):
    normalized = normalize(text)
    chars = [KEEP] * len(normalized.text)
    probs = [1.0] * len(normalized.text)
    for index, label, prob in edits:
        chars[index], probs[index] = label, prob
    return build_result(text, normalized, Prediction(KEEP, 1.0, chars, probs), threshold, calibration)


def test_corrections_have_spans_in_the_original_text():
    text = f"ذهبت ال{TATWEEL}ي الجامعه"  # the tatweel is removed by normalization
    source = normalize(text).text            # "ذهبت الي الجامعه"
    result = _result(text, [(source.index("الي") + 2, make_label(REPLACE, "ى"), 0.97),
                            (len(source) - 1, make_label(REPLACE, "ة"), 0.93)])
    assert result.text == "ذهبت الى الجامعة"
    first, second = result.corrections
    assert (first.original, first.replacement) == (f"ال{TATWEEL}ي", "الى")
    assert text[first.start:first.end] == f"ال{TATWEEL}ي" and first.confidence == 0.97
    assert (second.original, second.replacement, second.category) == ("الجامعه", "الجامعة", "ta_marbuta")
    assert "".join(segment for segment, _ in result.segments) == result.text


def test_a_merge_is_one_correction_over_both_words():
    text = "ثم و يستخدم هنا"
    result = _result(text, [(text.index(" يستخدم"), make_label(DELETE), 0.99)])
    (merge,) = result.corrections
    assert (merge.original, merge.replacement) == ("و يستخدم", "ويستخدم")


def test_unconfident_edits_are_not_applied_and_confidence_is_calibrated():
    text = "الي"
    assert _result(text, [(2, make_label(REPLACE, "ى"), 0.6)], threshold=0.9).corrections == []
    blocks = [{"up_to": 0.5, "probability": 0.3}, {"up_to": 1.0, "probability": 0.8}]
    assert _result(text, [(2, make_label(REPLACE, "ى"), 0.95)], calibration=blocks).corrections[0].confidence == 0.8
    assert calibrated(0.2, blocks) == 0.3
