"""Thresholding predictions: only confident edits of editable characters are applied."""
from araspellx.correct.decode import Prediction, at_threshold
from araspellx.data.labels import DELETE, KEEP, REPLACE, apply_labels, make_label


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
