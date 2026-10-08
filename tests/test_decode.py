"""Thresholding predictions: only confident, allowed edits are applied."""
from araspellx.correct.decode import Prediction, at_threshold
from araspellx.data.labels import DELETE, KEEP, REPLACE, apply_labels, make_label


def _apply(text, chars, threshold=0.5):
    prediction = Prediction(cls=KEEP, cls_prob=1.0, chars=chars, probs=[1.0] * len(text))
    return apply_labels(text, at_threshold(prediction, text, threshold))


def test_only_confident_edits_of_editable_characters_are_applied():
    text = "الي 5"
    prediction = Prediction(
        cls=make_label(KEEP, insert="و"), cls_prob=0.97,
        chars=[KEEP, KEEP, make_label(REPLACE, "ى"), KEEP, make_label(DELETE)],
        probs=[1.0, 1.0, 0.9, 1.0, 0.99])  # the digit's deletion must never apply
    assert apply_labels(text, at_threshold(prediction, text, 0.5)) == "والى 5"
    assert apply_labels(text, at_threshold(prediction, text, 0.95)) == "والي 5"
    assert apply_labels(text, at_threshold(prediction, text, 0.99)) == "الي 5"


def test_cls_can_only_insert():
    text = "ب"
    prediction = Prediction(cls=make_label(DELETE), cls_prob=1.0, chars=[KEEP], probs=[1.0])
    assert at_threshold(prediction, text, 0.5).cls == make_label(KEEP)


def test_spacing_next_to_punctuation_follows_typography():
    semicolon, comma = chr(0x061B), chr(0x060C)
    text = f"السهلة{semicolon} صنعاءء :"
    chars = [KEEP] * len(text)
    chars[5] = make_label(KEEP, insert=" ")  # a space before the semicolon: never added
    chars[7] = make_label(DELETE)            # the space after it: never removed
    chars[13] = make_label(REPLACE, comma)   # a letter that OCR made of a comma: restored
    chars[14] = make_label(DELETE)           # the space before the colon: removed
    assert _apply(text, chars) == f"السهلة{semicolon} صنعاء{comma}:"


def test_brackets_attach_inside():
    text = "قال ( نعم ) ثم"
    chars = [KEEP] * len(text)
    chars[5] = chars[9] = make_label(DELETE)  # spaces inside the brackets: removed
    chars[3] = chars[11] = make_label(DELETE)  # spaces outside: kept
    assert _apply(text, chars) == "قال (نعم) ثم"
