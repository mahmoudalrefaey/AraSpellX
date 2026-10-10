# Evaluation

AraSpellX is measured on frozen test sets of real text: real typing errors, real scans, classical books and clean text. Training data is filtered against all of them ([data](data.md)). This page describes the test sets, the metrics, the release gates, how to run the evaluation and the results of the current model.

## Test sets

| ID | Set | Content | Use |
|---|---|---|---|
| T-1 | Real typed errors | 7,627 Arabic Wikipedia paragraphs before and after one editor's spelling fixes (8,598 fixes), from held-out pages | CC BY-SA |
| T-4 | Real OCR, modern Arabic | 300 Yarmouk articles: real 300-dpi scans read by Tesseract (3,184 paragraphs) and the dataset's ABBYY output (3,237 paragraphs) | open |
| T-5 | Degraded scans | NOD: 100 Yarmouk pages in many noise versions. The builder is ready; the OCR has not been run yet | CC BY 4.0 |
| T-6 | Real OCR, classical Arabic | OpenITI gold-standard lines of seven classical books, five lines per paragraph, read by Tesseract and by OpenITI's OCR (1,430 paragraphs each) | evaluation only (CC BY-NC-SA 4.0) |
| T-7 | Clean text (damage) | held-out paragraphs: modern (50,009 words), classical (50,036), diacritized (29,505), dialect (45,036) | CC BY-SA |
| — | Benchmark | 27 typed sentences with hand-corrected references (`benchmarks/real_world.csv`) | this project |

The evaluation also uses development sets to choose the threshold and fit the calibration, never the test sets: 2,000 clean and 1,000 noisy held-out paragraphs, the development share of the OCR training pairs (1,966) and the Yarmouk development articles (7,597 paragraphs, both engines).

## Metrics

Scoring (`araspellx/eval/metrics.py`) works per source word, so merged and split words are handled: each word owns its region of the text (the word and the spaces after it), and an alignment shows what the reference and the model made of that region.

- **Precision, recall, F0.5**: over word edits; F0.5 weighs precision twice as much as recall. Precision counts exact fixes only: a word with two errors of which one is fixed counts as a wrong edit.
- **Harmful edits**: the share of the model's edits that changed a correct word or did not bring a wrong word closer to the reference. Partial fixes count as improvements, not harm.
- **Damage**: the share of correct words that the model changed.
- **CER / WER**: character and word error rates against the reference, before and after correction.
- **Error categories** (`araspellx/eval/categories.py`): hamza on alef, hamza seat, ta marbuta, alef maqsura, alef after waw, dots, spacing, typo, mixed.

References are not perfect: Wikipedia text and OCR ground truth contain spelling errors, so some edits counted as damage or harm fix the reference itself. T-1 is affected most, because a single edit fixes only some of a paragraph's errors. **T-1 is therefore scored on the words the editors fixed**: recall is the share of those words the model fixed the same way, precision the share of the model's changes to those words that match the editor. The model's other edits are counted per 1,000 words and checked against the fixes that editors made on training pages.

## Release gates

| Gate | Target |
|---|---|
| Damage on clean text (T-7) | ≤ 0.1% of words (modern), ≤ 0.2% (classical, diacritized), ≤ 0.5% (dialect) |
| Real typed errors (T-1) | precision ≥ 0.90 on the words editors fixed; recall and other edits reported |
| Real OCR, modern (T-4) | word errors down ≥ 20%; harmful edits ≤ 5% of the model's edits; no worse than raw OCR on ≥ 98% of pages |
| Real OCR, classical (T-6) and degraded scans (T-5) | no harm: error rates not higher |
| Speed | ≥ 300 words/s on a 4-core laptop CPU |
| Long documents | paragraph and page quality within 10% of sentence quality |
| Confidence | expected calibration error ≤ 0.05, after calibration on development data for each kind of input |

The OCR gate was first written as exact-fix precision ≥ 0.85. On OCR text about a third of the model's edits fix a word only partly, which exact-fix precision counts as wrong although the text improved; the gate therefore limits edits that make text worse, and exact-fix precision is still reported.

## Running the evaluation

```bash
python -m araspellx.eval.run --model artifacts/correct/best_model --out artifacts/eval/correct
```

The model reads every development and test text once; eight confidence thresholds (0.5 to 0.98) are then applied to the stored predictions. The threshold is the one with the highest mean F0.5 over the development error sets among thresholds that keep damage on clean development text within 0.05% (half the modern-text gate). The tool fits the calibration on development data, scores every test set at the chosen threshold and writes:

- `report.md`: gates, every set at every threshold, error categories on real scans, calibration and samples of harmful edits;
- `results.json`: all numbers;
- `calibration.json`: the threshold and the confidence calibration for OCR and typed input; copy it into the model folder.

A full run takes about 10 minutes on a laptop GPU. `--limit N` scores only the first N texts of each set for a quick check; `--threshold X` skips the threshold choice.

CPU speed is measured separately, on the model exported to ONNX:

```bash
python -m araspellx.eval.speed --configs 8x384 --threads 4
```

## Results of the current model

Model `artifacts/correct/best_model` (correction step 28,000), threshold 0.9:

| Gate | Target | Measured | |
|---|---|---|---|
| Damage, clean modern text | ≤ 0.1% | 0.058% | pass |
| Damage, clean classical text | ≤ 0.2% | 0.010% | pass |
| Damage, clean diacritized text | ≤ 0.2% | 0.020% | pass |
| Damage, dialect text | ≤ 0.5% | 0.007% | pass |
| Real typed errors (T-1) | precision ≥ 0.90 | 0.962 (recall 0.082) | pass |
| Real scans (T-4): word errors removed | ≥ 20% | 13.2% (WER 24.0% → 20.8%) | **fail** |
| Real scans (T-4): harmful edits | ≤ 5% | 5.1% (exact-fix precision 0.685) | **fail** |
| Real scans (T-4): pages no worse | ≥ 98% | 99.7% (589 of 591) | pass |
| Classical OCR (T-6): no harm | not higher | CER 12.6% → 12.3%, WER 43.3% → 41.9% | pass |
| Confidence calibration | ECE ≤ 0.05 | 0.048 (0.085 before calibration) | pass |
| Speed | ≥ 300 words/s | 410 (int8) / 337 (fp32) words/s | pass |
| Degraded scans (T-5), long documents | | | not measured yet |

| Set | Precision | Recall | Harmful edits | Damage | WER before → after |
|---|---|---|---|---|---|
| T-4, Tesseract | 0.536 | 0.058 | 6.4% | 0.06% | 20.5% → 18.9% |
| T-4, ABBYY | 0.738 | 0.182 | 4.7% | 0.19% | 27.4% → 22.7% |
| T-6, Tesseract | 0.689 | 0.037 | 10.7% | 0.07% | 39.1% → 37.6% |
| T-6, OpenITI OCR | 0.490 | 0.024 | 20.9% | 0.04% | 47.4% → 46.2% |
| Benchmark | 1.000 | 0.900 | 0.0% | 0.00% | 38.4% → 3.8% |

On T-1 the model fixed 706 of the 8,604 words the editors fixed, and 706 of the 734 of these words it changed match the editor. It made 3.4 other edits per 1,000 words, 76% of which are fixes that editors made on other pages (such as اللغه to اللغة and ايضا to أيضا).

**In short.** Typed spelling errors are corrected precisely, and correct, dialect and diacritized text is left alone. On real scans the model rarely makes a page worse, but it fixes too little: most OCR errors are badly garbled words (79% of the errors in T-4), and it fixes about 1 in 10 of them. Lowering the threshold does not help: on the development scans it adds fixes slowly while harmful edits rise quickly.
