# Architecture

## Task
Input: raw text, any length, any Unicode. Output: corrected text plus a list of edits (position in the original text, original, replacement, confidence, category).

## Text processing
- **Character set** (`araspellx/text/charset.py`): 210 tokens, covering Arabic letters (including alef wasla and Persian/Urdu letters used in foreign names), diacritics, tatweel, Arabic and ASCII punctuation, three digit systems, Latin letters, symbols, whitespace and five special tokens. Anything else is `[UNK]`.
- **Editable characters**: only Arabic letters, diacritics, tatweel and spaces may be changed. Digits, Latin text, punctuation and symbols are always kept.
- **Normalization** (`araspellx/text/normalize.py`): folds look-alike code points (ی → ي, ک → ك, ہ/ھ/ە → ه, ۃ → ة) and presentation-form ligatures, and removes tatweel and invisible characters. Every change is an edit, and every normalized character keeps a link to its original span, so later edits map back to the caller's text exactly. Diacritics are kept unless removal is requested.
- **Tokenizer** (`araspellx/text/tokenizer.py`): one token per character, a standard `PreTrainedTokenizerFast` (loadable with `AutoTokenizer`) with exact character offsets.

## Model
- Our own implementation of a BERT encoder (`araspellx/model/bert.py`): post-norm layers, GELU, learned absolute positions. Parameter names, shapes and computations match Hugging Face's `BertForMaskedLM` and `BertForTokenClassification`; `tests/test_bert.py` checks outputs agree within 1e-5 and that checkpoints load in both directions.
- Size: 8 layers, hidden size 384, 6 heads, feed-forward 1536, 512 positions, 14.6M parameters. Chosen as the largest size that meets the CPU speed target with refinement passes (`araspellx/eval/speed.py`, 4 threads, i5-10500H, ONNX int8):

  | Layers × hidden | Parameters | Words/s (fp32) | Words/s (int8) |
  |---|---|---|---|
  | 6 × 384 | 11.0M | 699 | 954 |
  | **8 × 384** | **14.6M** | **491** | **746** |
  | 12 × 384 | 21.7M | 360 | 492 |
  | 6 × 512 | 19.4M | 460 | 628 |
  | 8 × 512 | 25.7M | 331 | 499 |

## Edit labels
Each input character gets one label (`araspellx/data/labels.py`):

| Label | Meaning |
|---|---|
| `K` | keep |
| `D` | delete |
| `R:x` | replace with x |
| `…+s` | then insert the string s after the character |

The `[CLS]` token carries insertions before the first character. Labels are derived by aligning noisy and clean text character by character; the label set (about 160 labels) is the most frequent edits covering 99.5% of training edits. Why labels rather than generating the corrected text: one pass instead of one decoder step per character, correct text kept by default, and native offsets and confidences.

## Training
1. **Pretraining** (`araspellx/train/pretrain.py`): masked character prediction on Wikimedia text, 15% of characters per 512-token window, half as whole words and half as 1–5 character spans.
2. **Correction** (`araspellx/train/correct.py`): the pretrained encoder plus a label classifier, trained on a mixture of clean text, typed-error noise and OCR pairs (see [data](data.md)).

## Decoding
`araspellx/correct/decode.py` reads long text in 512-token windows with 64-character margins; each character takes the prediction of the window where it is most central. An edit is applied only if its probability reaches the threshold; non-editable characters are never changed. Refinement passes run the model again on the corrected text.
