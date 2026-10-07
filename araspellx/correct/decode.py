"""Turn model predictions into corrected text (used by evaluation and the helper).

Text is read in 512-token windows; each character takes the prediction of
the window in which it sits furthest from the edges (at least MARGIN
characters of context on both sides when the text is long enough). A
predicted edit is applied only if its probability reaches `threshold`;
non-editable characters (digits, Latin, punctuation...) are always kept.
Refinement passes run the model again on the corrected text.
"""
from __future__ import annotations

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


@torch.no_grad()
def predict_labels(model, texts: Sequence[str], vocab: LabelVocab, device,
                   threshold: float = 0.5, batch_size: int = 32) -> List[Tuple[Labels, List[float]]]:
    """Per-character labels (and the probability of each chosen edit) for every text."""
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
        probs = logits.float().softmax(-1)
        prob, label_id = probs.max(-1)
        for r, (t, s) in enumerate(part):
            n = min(CONTENT, len(texts[t]) - s)
            if s == 0:
                cls_labels[t] = (vocab.decode(label_id[r, 0].item()), prob[r, 0].item())
            for k in range(n):
                centrality = min(k, n - 1 - k)
                if centrality > best[t][s + k][0]:
                    best[t][s + k] = (centrality, vocab.decode(label_id[r, k + 1].item()),
                                      prob[r, k + 1].item())
    results = []
    for t, text in enumerate(texts):
        chars, probs = [], []
        for char, (_, label, p) in zip(text, best[t]):
            if label == KEEP or p < threshold or not is_editable(char):
                label = KEEP
            chars.append(label)
            probs.append(p)
        cls, p = cls_labels[t]
        cls = cls if p >= threshold and parse_label(cls)[0] == KEEP else make_label(KEEP)
        results.append((Labels(cls, chars), probs))
    return results


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
