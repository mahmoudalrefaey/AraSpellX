"""Turn model predictions into corrected text (used by evaluation and the helper).

Text is read in 512-token windows; each character takes the prediction of
the window in which it sits furthest from the edges (at least MARGIN
characters of context on both sides when the text is long enough). A
predicted edit is applied only if its probability reaches `threshold`;
non-editable characters (digits, Latin, punctuation...) are always kept.
Refinement passes run the model again on the corrected text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from araspellx.data.labels import KEEP, Labels, LabelVocab, apply_labels, make_label, parse_label
from araspellx.text.charset import CLS, PAD, SEP, TOKEN_TO_ID, is_editable
from araspellx.text.tokenizer import encode_chars

WINDOW = 512
CONTENT = WINDOW - 2
MARGIN = 64


def window_starts(n: int) -> List[int]:
    if n <= CONTENT:
        return [0]
    stride = CONTENT - 2 * MARGIN
    starts = list(range(0, n - CONTENT, stride))
    starts.append(n - CONTENT)
    return starts


@dataclass
class Prediction:
    """The most likely label of every character (and of [CLS]) with its probability."""
    cls: str
    cls_prob: float
    chars: List[str]
    probs: List[float]


def at_threshold(prediction: Prediction, text: str, threshold: float) -> Labels:
    """The edits whose probability reaches `threshold`; everything else is kept."""
    chars = [KEEP if label == KEEP or p < threshold or not is_editable(char) else label
             for char, label, p in zip(text, prediction.chars, prediction.probs)]
    cls = prediction.cls
    if prediction.cls_prob < threshold or parse_label(cls)[0] != KEEP:  # [CLS] can only insert
        cls = make_label(KEEP)
    return Labels(cls, chars)


def predict_labels(model, texts: Sequence[str], vocab: LabelVocab, device,
                   threshold: float = 0.5, batch_size: int = 32) -> List[Tuple[Labels, List[float]]]:
    """Per-character labels (and the probability of each chosen edit) for every text."""
    return [(at_threshold(p, text, threshold), p.probs)
            for p, text in zip(predict(model, texts, vocab, device, batch_size), texts)]


@torch.no_grad()
def predict(model, texts: Sequence[str], vocab: LabelVocab, device, batch_size: int = 32) -> List[Prediction]:
    """Run the model once over every text (see the module docstring for windows)."""
    jobs = []  # (text index, window start)
    for t, text in enumerate(texts):
        jobs += [(t, s) for s in window_starts(len(text))]
    best = [[(-1.0, KEEP, 1.0)] * len(text) for text in texts]  # (centrality, label, prob)
    cls_labels = [(KEEP, 1.0)] * len(texts)
    for i in range(0, len(jobs), batch_size):
        part = jobs[i:i + batch_size]
        rows = [[TOKEN_TO_ID[CLS]] + encode_chars(texts[t][s:s + CONTENT]) + [TOKEN_TO_ID[SEP]] for t, s in part]
        length = max(map(len, rows))
        ids = torch.full((len(rows), length), TOKEN_TO_ID[PAD], dtype=torch.long)
        for r, row in enumerate(rows):
            ids[r, :len(row)] = torch.tensor(row)
        ids = ids.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(ids, (ids != TOKEN_TO_ID[PAD]).long())
        prob, label_id = logits.float().softmax(-1).max(-1)
        prob, label_id = prob.cpu().tolist(), label_id.cpu().tolist()  # one transfer per batch
        for r, (t, s) in enumerate(part):
            n = min(CONTENT, len(texts[t]) - s)
            row_prob, row_label = prob[r], label_id[r]
            if s == 0:
                cls_labels[t] = (vocab.decode(row_label[0]), row_prob[0])
            for k in range(n):
                centrality = min(k, n - 1 - k)
                if centrality > best[t][s + k][0]:
                    best[t][s + k] = (centrality, vocab.decode(row_label[k + 1]), row_prob[k + 1])
    return [Prediction(cls_labels[t][0], cls_labels[t][1], [label for _, label, _ in best[t]],
                       [p for _, _, p in best[t]]) for t in range(len(texts))]


def correct_texts(model, texts: Sequence[str], vocab: LabelVocab, device,
                  threshold: float = 0.5, passes: int = 1) -> List[str]:
    """Corrected versions of already-normalized texts."""
    texts = list(texts)
    for _ in range(passes):
        predictions = predict_labels(model, texts, vocab, device, threshold)
        updated = [apply_labels(text, labels) for text, (labels, _) in zip(texts, predictions)]
        if updated == texts:
            break
        texts = updated
    return texts
