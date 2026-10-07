# Data

## Training text
Paragraphs extracted from the 2026-10-01 Wikimedia dumps (`araspellx/data/wikimedia.py`), with punctuation, numbers and Latin words kept:

| Source | Role | Pages with text |
|---|---|---|
| Arabic Wikipedia | Modern standard Arabic | 1,302,784 |
| Arabic Wikisource | Classical and diacritized text | 38,940 |
| Egyptian Arabic Wikipedia | Dialect ("leave alone"), capped at 150M characters | 1,614,151 |
| Moroccan Arabic Wikipedia | Dialect ("leave alone") | 20,332 |

All Wikimedia text is CC BY-SA.

Before training (`araspellx/data/pretrain_corpus.py`) paragraphs are normalized, exact duplicates removed, and any paragraph sharing an 8-word passage with any test set is dropped (`araspellx/data/dedup.py`).

## Held-out pages
`araspellx/data/splits.py` assigns each page to a bucket by hashing its wiki and page id:

| Buckets (of 10,000) | Use |
|---|---|
| 0–49 | Clean-text test sets (T-7) |
| 50–299 | Arabic Wikipedia pages whose real edits form the typed-error test set (T-1) |
| 300+ | Training |

## Noise sources for correction training
- **Clean text**, unchanged: teaches the model to leave correct text, dialect and diacritized text alone.
- **Typed-error noise** (`araspellx/noise/typed.py`): spelling habits measured in real text (dropped hamza on alef, ة/ه, ى/ي, hamza seats, ظ/ض, dropped alef after waw) and keyboard typos, merged and split words. Only Arabic words are touched.
- **Rendered OCR** (`araspellx/noise/ocr_pairs.py`): paragraphs rendered in open Arabic fonts and sizes, degraded like real scans (`araspellx/ocr/degrade.py`) and read back by Tesseract.
- **Real OCR**: Yarmouk scans read by Tesseract and the dataset's own ABBYY output, aligned to the ground truth paragraph by paragraph.
- **Real edits** (planned): spelling corrections mined from Arabic Wikipedia's edit history.

## OCR environment
`docker/ocr/Dockerfile` pins Ubuntu 24.04, Tesseract 5.3.4 with the `tessdata_best` 4.1.0 Arabic model, Pillow with libraqm for Arabic shaping, poppler and OFL fonts (Amiri, Noto Naskh/Sans/Kufi Arabic, Scheherazade). All OCR data, for training and testing, is produced in this environment.
