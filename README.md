<p align="center">
  <img src="assets/arasSpellX-banner.svg" alt="AraSpellX — Arabic Spelling Correction" width="100%">
</p>

# AraSpellX v0 (legacy baseline)

> **This branch is archived.** It contains v0 of AraSpellX, a reimplementation of the [AraSpell](https://arxiv.org/abs/2405.06981) character-level sequence-to-sequence approach. It is kept as the baseline that v1 is compared against and is no longer developed. Current work lives on the [`v1`](../../tree/v1) branch.

## What it is
A character-level Transformer encoder-decoder (about 20M parameters) that rewrites a noisy Arabic sentence into a corrected one. It is trained on Arabic Wikipedia sentences with synthetic errors: random typos (insertions, deletions, swaps, keyboard neighbours, merged words) and, in the later data version, spelling habits (dropped hamza, ة/ه, ى/ي, hamza seats, ظ/ض, dropped alef after waw).

| | |
|---|---|
| Model | Encoder-decoder, 4 + 4 layers, d_model 512, 8 heads, feed-forward 256 |
| Vocabulary | 40 tokens: 36 Arabic letters, space, PAD/SOS/EOS |
| Input length | Up to 128 characters per sentence (longer text is split into chunks by `inference.py`) |
| Training data | AraSpell corpus: 6.9M sentences from the Arabic Wikipedia 2021 dump |
| Decoding | Greedy, character by character |

## Setup
```bash
uv sync          # Python 3.10, PyTorch 2.3 (CUDA 12.1)
```
Download the data as described in [docs/DATA.md](docs/DATA.md). Training and evaluation need an NVIDIA GPU.

## Usage
```bash
# Train on the original synthetic errors (10% distortion)
python train.py --dist_key distorted_0.1

# Optional: add spelling habits and variable typo rates, then train on them
python -m data.regenerate_distortions --input data/dataset/train.csv
python -m data.regenerate_distortions --input data/dataset/test.csv
python train.py --train_path data/dataset/train_rw.csv --test_path data/dataset/test_rw.csv \
    --dist_key distorted_rw --outdir outdir_rw/ --logdir outdir_rw/logs

# Evaluate a checkpoint on the test set (CER/WER)
python evaluate.py --checkpoint outdir/<checkpoint>.pt

# Correct text, or score a CSV of (distorted, clean) pairs
python inference.py --checkpoint outdir/<checkpoint>.pt --text "ذهبت الي الجامعه"
python inference.py --checkpoint outdir/<checkpoint>.pt --eval_csv benchmarks/real_world.csv \
    --input_col distorted --ref_col clean

# Controlled typo/habit probes for comparing checkpoints
python evaluate_probes.py --checkpoint outdir/<checkpoint>.pt
```
Training saves a checkpoint every 5,000 steps and resumes with `--pre_trained_path <checkpoint>`.

## Results
Measured on an RTX 3060 laptop GPU.

| Model | Synthetic test (10% typos) | Real-world benchmark (27 sentences) |
|---|---|---|
| Input, no correction | CER 10.00%, WER 51.3% | CER 6.79%, WER 38.4% |
| 3 epochs on `distorted_0.1` (18.5 h) | CER 4.06%, WER 15.9% | CER 7.12%, WER 34.5% |
| 35k steps on `distorted_rw` (with habits) | — | CER 1.38%, WER 7.9% |

Controlled probes (3-epoch model): a word with one typo is restored 69% of the time, with two typos 39%, with three 30%.

The model learns the errors its generator produces well, but it does not generalize to real text. See [docs/LIMITATIONS.md](docs/LIMITATIONS.md).

## Files
| Path | Content |
|---|---|
| `train.py`, `core/train.py` | Training loop (mixed precision, gradient accumulation, resumable checkpoints) |
| `models/` | Encoder-decoder model, loss, optimizers |
| `data/` | Dataset, tokenizer, preprocessing and synthetic error generation |
| `evaluate.py`, `core/evaluate.py` | Test-set evaluation |
| `inference.py` | Correction of raw text: training-format padding, normalization, chunking, `--eval_csv` |
| `evaluate_probes.py` | Controlled corruption probes |
| `benchmarks/real_world.csv` | 27 hand-corrected real sentences |
| `docs/` | Data guide, limitations, performance notes |

## Credits
Based on AraSpell: M. Salhab and F. Abu-Khzam, *AraSpell: A Deep Learning Approach for Arabic Spelling Correction*, [arXiv:2405.06981](https://arxiv.org/abs/2405.06981), code at [msalhab96/AraSpell](https://github.com/msalhab96/AraSpell) (MIT).

## License
MIT. The training data is derived from Wikipedia (CC BY-SA).
