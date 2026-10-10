# Using AraSpellX

This guide covers installing AraSpellX and correcting text with a trained model: in the browser and from Python.

## Installation

Requirements: Python 3.10–3.12 and [uv](https://docs.astral.sh/uv/).

```bash
git clone -b v1 https://github.com/mahmoudalrefaey/AraSpellX.git
cd AraSpellX
uv sync          # creates .venv with PyTorch 2.3 (CUDA 12.1 wheels, which also run on CPU)
```

A GPU is optional for correcting text; the model is small enough for a CPU.

## The model

The released model is on Hugging Face as [`mahmoudalrefaey/AraSpellX`](https://huggingface.co/mahmoudalrefaey/AraSpellX). Wherever a model is expected you can give either that id, which is downloaded once (60 MB) and cached, or a local folder, such as a model you trained yourself ([training](training.md)). A model folder holds:

| File | Content |
|---|---|
| `config.json`, `model.safetensors` | the correction model, a standard Hugging Face `BertForTokenClassification` |
| `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` | the character tokenizer |
| `labels.json` | the edit labels the model predicts |
| `calibration.json` | confidence calibration and the default threshold, written by the evaluation ([evaluation](evaluation.md)); copy it into the folder |

Without `calibration.json` the model still works: confidences are the raw probabilities and the threshold is 0.9.

## Try it in the browser

The demo page runs online at **[araspellx.streamlit.app](https://araspellx.streamlit.app)**. It is `demo/streamlit_app.py`, a [Streamlit](https://streamlit.io) app that uses the Hugging Face model (downloaded on the first visit). Type or paste Arabic text, up to 5,000 characters, choose **نص مكتوب** (typed text) or **نص من مسح ضوئي** (OCR output), optionally change the confidence threshold, and press **صحّح**. The page shows the text before (errors in red) and after (corrections in green; hover for the original word and the confidence), and a table of every correction with its confidence and error type.

To run it on your computer from the project's environment (Streamlit is one of its dependencies):

```bash
uv run streamlit run demo/streamlit_app.py      # opens http://localhost:8501
```

Or in a fresh virtual environment (not the project's `.venv`, whose CUDA PyTorch it would replace), with the CPU-only PyTorch that Community Cloud uses:

```bash
pip install -r demo/requirements.txt
streamlit run demo/streamlit_app.py
```

## Publishing the demo on Streamlit Community Cloud

[Streamlit Community Cloud](https://share.streamlit.io) hosts the live demo for free, straight from the GitHub repository and without Docker. To publish it, or your own copy:

1. Push the branch with the `demo/` folder to GitHub.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub and choose **Create app** → **Deploy a public app from GitHub**.
3. Repository `mahmoudalrefaey/AraSpellX`, branch `v1`, main file path `demo/streamlit_app.py`; pick the app address. Under **Advanced settings**, choose Python 3.10, 3.11 or 3.12: PyTorch 2.3 and NumPy 1.26, pinned in `demo/requirements.txt`, have no wheels for later versions.
4. **Deploy.** The first build takes several minutes; later pushes to the branch update the app by themselves.

Community Cloud installs `demo/requirements.txt` (CPU-only PyTorch) rather than the repository's `uv.lock`. An app that has had no visitors for 12 hours goes to sleep, and the next visitor wakes it up. The text is processed on Streamlit's servers and is not saved.

## Python

```python
from araspellx.correct.corrector import Corrector

corrector = Corrector("mahmoudalrefaey/AraSpellX")         # or a local folder; device, threshold and calibration are optional
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

Merging two words is one correction covering both: `و يستخدم` becomes `ويستخدم`.

## What the model changes, and what it leaves alone

- **Only Arabic letters, diacritics, tatweel and spaces are edited.** Digits, Latin text and symbols are never changed. Existing punctuation is never changed; a letter that an OCR engine made of a punctuation mark may be turned back into it.
- **Spacing next to punctuation follows Arabic typography**: a mark attaches to the word before it (opening brackets and quotes to the word after); a space on that side may be removed, never added, and the space on the other side is never removed.
- **An edit is applied only above the confidence threshold** (0.9 for the current model, chosen on development data). When the model is unsure, the word stays as written.
- **The input is normalized first**: look-alike Persian and Urdu letters are folded into Arabic ones (ی becomes ي and ک becomes ك), presentation-form ligatures are expanded, letters stored as a base letter plus a combining hamza or madda are composed, and tatweel and invisible characters are removed. `Result.text` is this normalized text with the corrections applied; correction spans always refer to the original input.

## Target spelling and scope

The target is **standard modern Arabic orthography**, as in edited publications and Arabic Wikipedia: hamzat qat' written (أ, إ, آ), hamzat wasl written as a bare alef (استخدام, الاستفادة), standard ta marbuta and alef maqsura. Older spellings such as فى and الى are modernized, as Wikipedia's spelling bots do.

Out of scope: grammar (agreement, case endings, verb moods), punctuation and style, converting dialect to standard spelling, adding or fixing diacritics, and OCR itself (handwriting, layout, reading order).

## Speed

On a laptop CPU (Intel i5-10500H, 4 threads), the model exported to ONNX processes **337 words/s** (fp32) or **410 words/s** (int8), measured with `python -m araspellx.eval.speed --configs 8x384 --threads 4`. The `Corrector` runs PyTorch: on the same CPU it loads the model in under a second, corrects a sentence in about 20 ms and long text at about 270–300 words/s (4–6 threads). ONNX inference in the helper package is planned.

## Limitations

Measured on the frozen test sets ([evaluation](evaluation.md)):

- **Typed text**: corrections are reliable (precision 0.96 on real Wikipedia spelling fixes, 1.00 on the 27 hand-corrected benchmark sentences), but the model fixes only the errors it is sure about.
- **OCR text**: the model rarely makes a page worse (99.7% of real scanned pages are no worse), but it removes only 13% of word errors on real scans (17% on ABBYY output, 8% on Tesseract output): badly garbled words are mostly left alone.
- **Errors that produce another valid word** need meaning rather than spelling, which a 15M-parameter character model knows little about.
- **Quranic text and accepted spelling variants are not protected yet**: verses are corrected like any other text, and either form of a word with two accepted spellings (such as مسؤول and مسئول, or مائة and مئة) may be changed.
- Dialect and diacritized text are left alone in tests (damage 0.007% and 0.020% of words), but the model does not correct them.
