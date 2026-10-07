# Evaluation

## Test sets
All test sets are fixed before training, and training data is filtered against them.

| ID | Set | Content | License use |
|---|---|---|---|
| T-1 | Real typed errors | Spelling corrections mined from Arabic Wikipedia's edit history (held-out pages). In progress | CC BY-SA |
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
- **Precision**, **recall** and **F0.5** (precision weighted higher) over edits.
- **Damage**: the share of correct words the output changed.
- **CER/WER** before and after correction.
- Per-category counts (`araspellx/eval/categories.py`): hamza on alef, hamza seat, ta marbuta, alef maqsura, alef after waw, dots, spacing, typo.

## Release gates (targets for v1)

| Gate | Target |
|---|---|
| Damage on clean text | ≤ 0.1% of words (modern), ≤ 0.2% (classical, diacritized), ≤ 0.5% (dialect) |
| Real typed errors | Edit precision ≥ 0.90; recall reported |
| Real OCR, modern | Word error rate down ≥ 20%; edit precision ≥ 0.85; no worse than raw OCR on ≥ 98% of pages |
| Real OCR, classical and degraded scans | No harm |
| Speed | ≥ 300 words/s on a 4-core laptop CPU |
| Long documents | Paragraph and page quality within 10% of sentence quality |
| Confidence | Expected calibration error ≤ 0.05 |
