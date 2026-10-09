# Using AraSpellX

This guide covers installing AraSpellX and correcting text with a trained model: in the browser, on the command line and from Python.

## Installation

Requirements: Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone -b v1 https://github.com/mahmoudalrefaey/AraSpellX.git
cd AraSpellX
uv sync          # creates .venv with PyTorch 2.3 (CUDA 12.1 wheels, which also run on CPU)
```

A GPU is optional for correcting text; the model is small enough for a CPU.

## The model folder

AraSpellX is not published on Hugging Face yet, so you need a model trained with this repository ([training](training.md)). A trained model is a folder like `artifacts/correct/best_model`:

| File | Content |
|---|---|
| `config.json`, `model.safetensors` | the correction model, a standard Hugging Face `BertForTokenClassification` |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | the character tokenizer |
| `labels.json` | the edit labels the model predicts |
| `calibration.json` | confidence calibration and the default threshold, written by the evaluation ([evaluation](evaluation.md)); copy it into the folder |

Without `calibration.json` the model still works: confidences are the raw probabilities and the threshold is 0.9.

## Try it in the browser

```bash
python -m araspellx.correct.demo
```

This starts a small web page at **http://localhost:8000** and opens it. Type or paste Arabic text, choose **نص مكتوب** (typed text) or **نص من مسح ضوئي** (OCR output) and press **صحّح**. The page shows the text before (errors in red) and after (corrections in green; hover for the original word and the confidence), and a table of every correction with its confidence and error type. The page runs locally: the text never leaves the computer. Stop it with Ctrl+C.

## Command line

```bash
python -m araspellx.correct.demo --text "ذهبت الي الجامعه"   # correct one text
python -m araspellx.correct.demo --cli                       # correct texts one after another
```

Output for the sentence used in the demo's first example (model of the current evaluation):

```text
ذهبت إلى الجامعة صباحا لكي أحضر المحاضرة الأولى، ثم قابلت صديقي في المكتبة.
  الي -> إلى   (98%, mixed, characters 5-8)
  الجامعه -> الجامعة   (100%, ta_marbuta, characters 9-16)
  احضر -> أحضر   (94%, hamza_alef, characters 27-31)
  ...
```

| Option | Meaning |
|---|---|
| `--model PATH` | model folder (default `artifacts/correct/best_model`) |
| `--source typed\|ocr` | kind of input; selects the confidence calibration (default `typed`) |
| `--threshold X` | minimum confidence for an edit (default: the one in `calibration.json`, else 0.9) |
| `--device cpu\|cuda` | default: CUDA when available |
| `--port N`, `--no_browser` | web page settings |

Terminals often display Arabic left to right or with disconnected letters; the web page shows it correctly.

## Python

```python
from araspellx.correct.corrector import Corrector

corrector = Corrector("artifacts/correct/best_model")      # device, threshold and calibration are optional
result = corrector.correct("ذهبت الي الجامعه", source="typed")

result.text            # the corrected text
for c in result.corrections:
    print(c.start, c.end, c.original, "->", c.replacement, round(c.confidence, 2), c.category)
```

| Field | Meaning |
|---|---|
| `Result.text` | the corrected text, in normalized form (see below) |
| `Result.corrections` | one `Correction` per changed word, in text order |
| `Result.segments` | the corrected text split into pieces, each with its `Correction` or `None`, for highlighting |
| `Correction.start`, `.end` | span of the original word in **your input** text |
| `Correction.original`, `.replacement` | the word before and after |
| `Correction.confidence` | calibrated probability that the correction is right (the lowest over the word's edited characters) |
| `Correction.category` | `hamza_alef`, `hamza_seat`, `ta_marbuta`, `alef_maqsura`, `alef_fariqa`, `dots`, `spacing`, `typo` or `mixed` |

Merging two words is one correction covering both (`و يستخدم` → `ويستخدم`).

## What the model changes, and what it leaves alone

- **Only Arabic letters, diacritics, tatweel and spaces are edited.** Digits, Latin text and symbols are never changed. Existing punctuation is never changed; a letter that an OCR engine made of a punctuation mark may be turned back into it.
- **Spacing next to punctuation follows Arabic typography**: a mark attaches to the word before it (opening brackets and quotes to the word after); a space on that side may be removed, never added, and the space on the other side is never removed.
- **An edit is applied only above the confidence threshold** (0.9 for the current model, chosen on development data). When the model is unsure, the word stays as written.
- **The input is normalized first**: look-alike Persian and Urdu letters are folded into Arabic ones (ی → ي, ک → ك), presentation-form ligatures are expanded, letters stored as a base letter plus a combining hamza or madda are composed, and tatweel and invisible characters are removed. `Result.text` is this normalized text with the corrections applied; correction spans always refer to the original input.

## Target spelling and scope

The target is **standard modern Arabic orthography**, as in edited publications and Arabic Wikipedia: hamzat qat' written (أ, إ, آ), hamzat wasl written as bare ا (استخدام, الاستفادة), standard ta marbuta and alef maqsura. Older spellings such as فى and الى are modernized, as Wikipedia's spelling bots do.

Out of scope: grammar (agreement, case endings, verb moods), punctuation and style, converting dialect to standard spelling, adding or fixing diacritics, and OCR itself (handwriting, layout, reading order).

## Speed

On a laptop CPU (Intel i5-10500H, 4 threads), the model exported to ONNX processes **337 words/s** (fp32) or **410 words/s** (int8), measured with `python -m araspellx.eval.speed --configs 8x384 --threads 4`. The demo and the `Corrector` run PyTorch; ONNX inference in the helper package is planned.

## Limitations

Measured on the frozen test sets ([evaluation](evaluation.md)):

- **Typed text**: corrections are reliable (precision 0.96 on real Wikipedia spelling fixes, 1.00 on the 27 hand-corrected benchmark sentences), but the model fixes only the errors it is sure about.
- **OCR text**: the model rarely makes a page worse (99.7% of real scanned pages are no worse), but it removes only 13% of word errors on real scans (17% on ABBYY output, 8% on Tesseract output): badly garbled words are mostly left alone.
- **Errors that produce another valid word** need meaning rather than spelling, which a 15M-parameter character model knows little about.
- **Quranic text and accepted spelling variants are not protected yet**: verses are corrected like any other text, and either form of a word with two accepted spellings (مسؤول/مسئول, مائة/مئة) may be changed.
- Dialect and diacritized text are left alone in tests (damage 0.007% and 0.020% of words), but the model does not correct them.
