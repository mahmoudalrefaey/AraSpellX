"""Correct text with a trained model: the corrected text plus every correction.

Each correction has its span in the caller's text, the original and the new
word, a confidence (calibrated for the kind of input when the model folder
holds a calibration.json written by araspellx.eval.run) and an error
category. Corrections are reported per word; merging two words is one
correction covering both.
"""
from __future__ import annotations

import json
import re
import string
from bisect import bisect_left
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from araspellx.correct.decode import Prediction, at_threshold, predict
from araspellx.data.labels import KEEP, LabelVocab, apply_labels
from araspellx.eval.categories import categorize
from araspellx.eval.metrics import regions_in, word_regions
from araspellx.model.bert import BertForTokenClassification
from araspellx.text.normalize import Normalized, normalize


PUNCTUATION = string.punctuation + "،؛؟«»…"


@dataclass
class Correction:
    start: int           # span [start, end) in the input text
    end: int
    original: str
    replacement: str
    confidence: float
    category: str


@dataclass
class Result:
    text: str                                    # the corrected text (normalized)
    corrections: List[Correction] = field(default_factory=list)
    segments: List[Tuple[str, Optional[Correction]]] = field(default_factory=list)  # for display, in order


def calibrated(probability: float, blocks: Sequence[Dict]) -> float:
    """Map a raw confidence through an isotonic calibration (blocks of up_to / probability)."""
    index = bisect_left([block["up_to"] for block in blocks], probability)
    return blocks[min(index, len(blocks) - 1)]["probability"]


def build_result(text: str, normalized: Normalized, prediction: Prediction, threshold: float,
                 calibration: Optional[Sequence[Dict]] = None) -> Result:
    """Corrected text and corrections of `text` from the model's prediction on its normalized form."""
    source = normalized.text
    labels = at_threshold(prediction, source, threshold)
    output = apply_labels(source, labels)
    regions = word_regions(source)
    outputs = regions_in(source, output, regions)
    result = Result(output)
    i = 0
    while i < len(regions):
        (a, b), new = regions[i], outputs[i]
        if new == source[a:b]:
            result.segments.append((new, None))
            i += 1
            continue
        if source[a:b].rstrip() == new.rstrip() and i + 1 < len(regions):  # a merge: show both words
            b, new = regions[i + 1][1], new + outputs[i + 1]
            i += 1
        probabilities = [prediction.probs[k] for k in range(a, b) if labels.chars[k] != KEEP]
        if a == 0 and labels.cls != KEEP:
            probabilities.append(prediction.cls_prob)
        confidence = min(probabilities) if probabilities else 1.0
        if calibration:
            confidence = calibrated(confidence, calibration)
        word = source[a:b].rstrip()
        start, end = normalized.to_original(a, a + len(word))
        correction = Correction(start, end, text[start:end], new.rstrip(), confidence,
                                categorize(word.strip(PUNCTUATION), new.rstrip().strip(PUNCTUATION)))
        result.corrections.append(correction)
        result.segments.append((new, correction))
        i += 1
    return result


HUB_MODEL = "mahmoudalrefaey/AraSpellX"  # the released model on Hugging Face
REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


class Corrector:
    """A trained correction model ready to correct text.

    corrector = Corrector("mahmoudalrefaey/AraSpellX")  # or a local folder such as artifacts/correct/best_model
    result = corrector.correct("ذهبت الي الجامعه", source="typed")  # or source="ocr"
    """

    def __init__(self, model: Union[str, Path] = HUB_MODEL, device: Optional[str] = None,
                 threshold: Optional[float] = None, calibration: Optional[Union[str, Path]] = None) -> None:
        """`model` is a local model folder or a Hugging Face model id (downloaded once, then cached)."""
        model_dir = Path(model)
        if not model_dir.exists() and REPO_ID.match(str(model)):
            from huggingface_hub import snapshot_download
            try:
                model_dir = Path(snapshot_download(str(model)))
            except Exception as error:  # unknown id, private repository, no network
                raise FileNotFoundError(f"could not download '{model}' from Hugging Face "
                                        f"({type(error).__name__}); check the model id and the connection") from error
        if not (model_dir / "config.json").is_file():
            raise FileNotFoundError(f"no model in '{model}': expected a folder with config.json, "
                                    "model.safetensors and labels.json, or a Hugging Face model id "
                                    f"such as {HUB_MODEL}")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = BertForTokenClassification.from_pretrained(model_dir).to(self.device).eval()
        self.vocab = LabelVocab.load(model_dir / "labels.json")
        path = Path(calibration) if calibration else model_dir / "calibration.json"
        self.calibration = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        self.threshold = threshold if threshold is not None else self.calibration.get("threshold", 0.9)

    def correct(self, text: str, source: str = "typed") -> Result:
        """Correct one text; `source` ("typed" or "ocr") selects the confidence calibration."""
        normalized = normalize(text)
        prediction = predict(self.model, [normalized.text], self.vocab, self.device)[0]
        return build_result(text, normalized, prediction, self.threshold, self.calibration.get(source))
