# Architecture

## Task
Input: raw text, any length, any Unicode. Output: corrected text plus a list of edits (position in the original text, original, replacement, confidence, category).

## Text processing
- **Character set** (`araspellx/text/charset.py`): 210 tokens, covering Arabic letters (including alef wasla and Persian/Urdu letters used in foreign names), diacritics, tatweel, Arabic and ASCII punctuation, three digit systems, Latin letters, symbols, whitespace and five special tokens. Anything else is `[UNK]`.
- **Editable characters**: only Arabic letters, diacritics, tatweel and spaces may be changed. Digits, Latin text, punctuation and symbols are always kept; a letter that OCR made of a punctuation mark may be turned back into it. Spacing next to punctuation follows Arabic typography: a mark attaches to the word before it (opening brackets and quotes to the word after), so a space on that side may be removed but never added, and the space on the other side is never removed (`allowed_label` in `araspellx/data/labels.py`, applied to training labels and predictions alike).
- **Normalization** (`araspellx/text/normalize.py`): folds look-alike code points (ی → ي, ک → ك, ہ/ھ/ە → ه, ۃ → ة) and presentation-form ligatures, composes letters stored as a base letter plus a combining hamza or madda (ا + ٔ → أ), and removes tatweel and invisible characters. Every change is an edit, and every normalized character keeps a link to its original span, so later edits map back to the caller's text exactly. Diacritics are kept unless removal is requested.
- **Tokenizer** (`araspellx/text/tokenizer.py`): one token per character, a standard `PreTrainedTokenizerFast` (loadable with `AutoTokenizer`) with exact character offsets.

## Model
- Our own implementation of a BERT encoder (`araspellx/model/bert.py`): post-norm layers, GELU, relative positions. Parameter names, shapes and computations match Hugging Face's `BertForMaskedLM` and `BertForTokenClassification`; `tests/test_bert.py` checks outputs agree within 1e-5 and that checkpoints load in both directions.
- **Relative positions** (Hugging Face's `position_embedding_type="relative_key"`): each attention score gets a learned term for the distance between the two characters. With BERT's usual absolute positions, character-level masked-LM pretraining stalled: the model learned only character frequencies and never learned to look at neighbouring characters (on real text, and on a toy task where every masked letter is given away by its neighbour). With relative distances it learns this within the first ~1,000 steps.
- Size: 8 layers, hidden size 384, 6 heads, feed-forward 1536, 512 positions, 15.1M parameters. Chosen by the CPU speed benchmark (`araspellx/eval/speed.py`, our exported model, 4 threads, i5-10500H, ONNX):

  | Layers × hidden | Positions | Parameters | Words/s (fp32) | Words/s (int8) |
  |---|---|---|---|---|
  | 6 × 384 | absolute | 11.0M | 699 | 954 |
  | 8 × 384 | absolute | 14.6M | 491 | 746 |
  | **8 × 384** | **relative_key** | **15.1M** | **337** | **410** |
  | 8 × 384 | relative_key_query | 15.1M | 245 | 268 |
  | 12 × 384 | absolute | 21.7M | 360 | 492 |
  | 6 × 512 | absolute | 19.4M | 460 | 628 |
  | 8 × 512 | absolute | 25.7M | 331 | 499 |

  The released checkpoints also load in plain `transformers`, whose relative-position code is slower on CPU; the helper package runs the faster ONNX export.

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
1. **Pretraining** (`araspellx/train/pretrain.py`): masked character prediction on Wikimedia text, 15% of the characters of each window, half as whole words and half as 1–5 character spans. As in the original BERT, the first 90% of steps use short windows (128 characters) and the rest full 512-character windows; every step sees 16,384 characters. The first full run collapsed at ~45k steps from attention logit growth: one head's query and key weights grew without limit until its scores reached 10 million. Training now caps every head's largest attention score at 50 by scaling its query (`araspellx/train/stability.py`, exact and without measurable cost), keeps the checkpoint with the best validation loss, and stops itself if validation loss gets 10% worse than the best.
2. **Correction** (`araspellx/train/correct.py`): the pretrained encoder plus a label classifier, trained on a mixture of clean text, typed-error noise and OCR pairs (see [data](data.md)).

## Decoding
`araspellx/correct/decode.py` reads long text in 512-token windows with 64-character margins; each character takes the prediction of the window where it is most central. An edit is applied only if its probability reaches the threshold; non-editable characters are never changed. Refinement passes run the model again on the corrected text.
