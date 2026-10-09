# Evaluation

## Test sets
All test sets are fixed before training, and training data is filtered against them.

| ID | Set | Content | License use |
|---|---|---|---|
| T-1 | Real typed errors | 7,627 paragraphs before and after spelling fixes from Arabic Wikipedia's edit history (8,598 fixes on held-out pages; `araspellx/testsets/edits.py`) | CC BY-SA |
| T-4 | Real OCR, modern Arabic | 300 articles of the [Yarmouk](https://www.kaggle.com/datasets/eyadwin/yarmouk-ocr-dataset) Testing split: real 300-dpi scans read by Tesseract (3,184 paragraphs) and the dataset's ABBYY output (3,237 paragraphs) | Open |
| T-5 | Degraded scans | [NOD](https://zenodo.org/records/5068735): 100 Yarmouk pages in many noise versions | CC BY 4.0 |
| T-6 | Real OCR, classical Arabic | [OpenITI](https://github.com/OpenITI/OCR_GS_Data) gold-standard lines of seven classical books, five lines per paragraph (1,430 paragraphs per engine) | Evaluation only (CC BY-NC-SA 4.0) |
| T-7 | Clean text (damage) | Held-out paragraphs: modern (50k words), classical (50k), dialect (45k), diacritized (29.5k) | CC BY-SA |
| — | Real-world benchmark | 27 typed sentences with references (`benchmarks/real_world.csv`) | Ours |

QALB and ZAEBUC may be reported as evaluation only (research licenses).

Raw OCR error rates before any correction:

| Set | Engine | CER | WER |
|---|---|---|---|
| T-4 | Tesseract | 9.1% | 21.3% |
| T-4 | ABBYY | 6.9% | 27.6% |
| T-6 | Tesseract | 15.9% | 48.9% |
| T-6 | OpenITI OCR | 14.6% | 49.2% |

## Metrics
Scoring (`araspellx/eval/metrics.py`) works per source word region (a word plus the spaces after it), so merged and split words are handled:

- **Gold edit**: the reference changes the region. **System edit**: the output changes it. **Correct**: the output equals the reference where an edit was needed.
- **Precision**, **recall** and **F0.5** (precision weighted higher) over edits. Precision counts exact fixes only: a word with two errors of which one is fixed counts as a wrong edit.
- **Harmful edits**: the share of system edits that changed a correct word or did not bring a wrong word closer to the reference. Edits that fix a word partly count as *improved*, not harmful.
- **Damage**: the share of correct words the output changed.

References are not perfect (Wikipedia text and OCR ground truth contain spelling errors), so some edits counted as damage or harmful are fixes of the reference itself.

T-1 is scored on the words the editors fixed: a single edit fixes only some of a paragraph's errors and leaves the others in the reference. The model's other edits are counted per 1,000 words and checked against the fixes editors made on training pages (`wiki_edits_train.jsonl`); clean-text damage (T-7) measures false corrections.

## Running the evaluation

```bash
python -m araspellx.eval.run --model artifacts/correct/model --out artifacts/eval/correct
```

The model reads every development and test text once; each confidence threshold (0.5 to 0.98) is then applied to the stored predictions. The threshold is chosen on development data only: the highest mean F0.5 over the development error sets among thresholds that keep damage on clean development text within 0.05% (half the modern-text gate). The frozen test sets are scored at that threshold. `report.md` lists the release gates, every set at every threshold, error categories on real scans, confidence calibration and samples of harmful edits; `results.json` holds all numbers.
- **CER/WER** before and after correction.
- Per-category counts (`araspellx/eval/categories.py`): hamza on alef, hamza seat, ta marbuta, alef maqsura, alef after waw, dots, spacing, typo.

## Release gates (targets for v1)

| Gate | Target |
|---|---|
| Damage on clean text | ≤ 0.1% of words (modern), ≤ 0.2% (classical, diacritized), ≤ 0.5% (dialect) |
| Real typed errors | Edit precision ≥ 0.90 on the words editors fixed; recall and the model's other edits reported |
| Real OCR, modern | Word error rate down ≥ 20%; harmful edits ≤ 5% of the model's edits (exact-fix precision reported); no worse than raw OCR on ≥ 98% of pages |
| Real OCR, classical and degraded scans | No harm |
| Speed | ≥ 300 words/s on a 4-core laptop CPU |
| Long documents | Paragraph and page quality within 10% of sentence quality |
| Confidence | Expected calibration error ≤ 0.05, after a confidence map fitted on development data for each kind of input (OCR, typed) |

The OCR gate was first written as exact-fix precision ≥ 0.85. On OCR text about a third of the model's edits fix a word only partly (one of two wrong letters), which exact-fix precision counts as wrong although the text improved; the gate therefore limits the edits that make text worse, and exact-fix precision is still reported.
