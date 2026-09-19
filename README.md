<p align="center">
  <img src="assets/arasSpellX-banner.svg" alt="AraSpellX — Arabic Spelling Correction" width="100%">
</p>

<p align="center">
  <a href="https://github.com/mahmoudalrefaey/AraSpellX/stargazers">
    <img src="https://img.shields.io/github/stars/mahmoudalrefaey/AraSpellX?style=for-the-badge&logo=github&label=Stars" alt="GitHub stars">
  </a>
  <a href="https://github.com/mahmoudalrefaey/AraSpellX">
    <img src="https://img.shields.io/github/languages/top/mahmoudalrefaey/AraSpellX?style=for-the-badge&label=Python 3.10" alt="Python">
  </a>
  <a href="https://arxiv.org/abs/2405.06981">
    <img src="https://img.shields.io/badge/Paper-AraSpell-8b5cf6?style=for-the-badge&logo=arxiv" alt="AraSpell paper">
  </a>
  <a href="https://huggingface.co/">
  <img src="https://img.shields.io/badge/Hugging%20Face-Model-yellow?style=for-the-badge" alt="Hugging Face">
  </a>
  <img src="https://img.shields.io/badge/Architecture-Transformer-purple?style=for-the-badge" alt="Transformer">

</p>

<h1 align="center">AraSpellX</h1>

<p align="center">
  <strong>Arabic spelling correction with Transformer-based sequence-to-sequence learning</strong><br>
  An independent Transformer implementation inspired by the AraSpell research work.
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#installation">Installation</a> •
  <a href="#training">Training</a> •
  <a href="#evaluation">Evaluation</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

Arabic text can contain character substitutions, missing characters, extra characters, keyboard noise, and context-dependent spelling errors.

**AraSpellX** is an independent implementation inspired by **AraSpell: A Deep Learning Approach for Arabic Spelling Correction** by Mahmoud Salhab and Faisal Abu-Khzam. The original research explores attentional RNN and Transformer sequence-to-sequence architectures with synthetic error generation for scalable Arabic spelling-correction training.

> **Note:** AraSpellX is not the official AraSpell repository. The original implementation is maintained at [msalhab96/AraSpell](https://github.com/msalhab96/AraSpell).

### Goals

- Build a reproducible Arabic spelling-correction training pipeline
- Support sequence-to-sequence correction of noisy Arabic text
- Generate synthetic spelling errors from clean Arabic data
- Evaluate models using CER and WER
- Make the implementation suitable for future Hugging Face releases
- Keep experiments transparent and reproducible

---

## Why Arabic spelling correction?

| Error type | Example |
|---|---|
| Missing character | `الجامعه` → `الجامعة` |
| Character substitution | `مسؤل` → `مسؤول` |
| Extra characters | `المدرسسسة` → `المدرسة` |
| Input / keyboard noise | Arabic character substitutions |
| Contextual errors | A valid word used incorrectly in context |

The goal is not simply to find a word in a dictionary. The model learns a mapping from a noisy sequence to its intended sequence.

---

## Architecture

AraSpellX focuses exclusively on the **Transformer-based sequence-to-sequence architecture**.

The original AraSpell project also includes RNN-based architectures, but those architectures are **not implemented or supported in AraSpellX**. They are mentioned only as part of the original research context.

```mermaid
flowchart LR
    A["Clean Arabic text"] --> B["Error Injection"]
    B --> C["Noisy Arabic text"]
    C --> D["Tokenizer"]
    D --> E["Transformer Encoder"]
    E --> F["Transformer Decoder"]
    F --> G["Corrected Arabic text"]
    G --> H["CER / WER"]
```

---

## Error generation

The reference research uses synthetic error generation to turn large amounts of clean Arabic text into training pairs. The paper reports experiments trained on more than **6.9 million Arabic sentences**.

```mermaid
flowchart LR
    A["Clean sentence"] --> B["Error generator"]
    B --> C["Insertion"]
    B --> D["Deletion"]
    B --> E["Substitution"]
    B --> F["Mixed corruption"]
    C --> G["Noisy sentence"]
    D --> G
    E --> G
    F --> G
```

Example:

```text
Clean:     اللغة العربية لغة جميلة
Distorted: اللغه العربيه لغة جميله
Target:    اللغة العربية لغة جميلة
```

---

## End-to-end pipeline

```mermaid
flowchart TB
    A["Arabic corpus"] --> B["Cleaning & normalization"]
    B --> C["Train / Dev / Test"]
    C --> D["Synthetic error injection"]
    D --> E["Tokenization"]
    E --> F["Model training"]
    F --> G["Validation"]
    G --> H["Checkpoint"]
    H --> I["Inference"]
    I --> J["Corrected Arabic"]
```

---

## Installation

### Clone

```bash
git clone https://github.com/mahmoudalrefaey/AraSpellX.git
cd AraSpellX
```

### Environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux / macOS:

```bash
source .venv/bin/activate
```

### Dependencies

```bash
pip install -r requirements.txt
```

For GPU training, install a PyTorch build compatible with the CUDA version available on your machine.

---

## Dataset format

The training pipeline uses paired clean and distorted text:

```text
clean,distorted
اللغة العربية جميلة,اللغه العربيه جميله
هذا مثال آخر,هاذا مثال اخر
```

Where:

- **clean** is the target sentence
- **distorted** is the noisy input

Keep train, development, and test data separated to avoid leakage.

---

## Training

The repository is being developed around a reproducible training workflow.

Typical usage:

```bash
python train.py \
    --epochs <epochs> \
    --train_path <train.csv> \
    --test_path <test.csv>
```

Before a full GPU run, validate:

- dataset integrity
- tokenizer and vocabulary
- maximum sequence length
- padding and special tokens
- train/dev/test separation
- GPU memory usage
- checkpoint creation
- validation metrics

---

## Inference

The intended inference flow is:

```text
Noisy Arabic text
        ↓
     Tokenizer
        ↓
    AraSpellX
        ↓
Corrected Arabic text
```

Example:

```text
Input:
الطلاب يدرسون اللغه العربيه

Output:
الطلاب يدرسون اللغة العربية
```

The final inference API will follow the implementation exposed by the current release.

---

## Evaluation

### Character Error Rate

**CER** measures character-level edits between the prediction and reference.

`CER ↓ = lower is better`

### Word Error Rate

**WER** measures word-level errors.

`WER ↓ = lower is better`

```mermaid
flowchart LR
    A["Predictions"] --> B["Normalization"]
    B --> C["Character comparison"]
    B --> D["Word comparison"]
    C --> E["CER"]
    D --> F["WER"]
```

---

## Results

### Reference benchmark

The table below contains selected results reported by the **original AraSpell research project**.

These are **reference results from AraSpell, not AraSpellX results**.

| Model | CER @ 5% | CER @ 10% | WER @ 5% | WER @ 10% |
|---|---:|---:|---:|---:|
| Transformer 0.05 | 1.24% | 4.15% | 5.35% | 18.38% |
| Transformer 0.1 | 1.45% | 2.82% | 5.95% | 10.36% |
| Transformer mixed | 1.11% | 2.80% | 4.80% | 10.65% |
| Transformer varied | 1.22% | 3.16% | 5.41% | 12.35% |

### AraSpellX benchmark

AraSpellX results will be added after the Transformer implementation and training pipeline are fully validated.

---

## Project status

**Development**

- [x] Repository initialized
- [x] Project documentation
- [x] AraSpellX implementation
- [ ] Dataset validation
- [ ] Full GPU training
- [ ] CER / WER benchmark
- [ ] Inference examples
- [ ] Hugging Face model release
- [ ] Reproducibility report

---

## Research reference

### AraSpell: A Deep Learning Approach for Arabic Spelling Correction

**Authors:** Mahmoud Salhab, Faisal Abu-Khzam

**Paper:** https://arxiv.org/abs/2405.06981

The paper presents an Arabic spelling-correction framework using RNN and Transformer Seq2Seq architectures with artificial error generation.

**Original implementation:**  
https://github.com/msalhab96/AraSpell

---

## Attribution

AraSpellX is an independent implementation focused on the **Transformer architecture** described in the AraSpell research work.

The project was developed with reference to:

**AraSpell — Arabic Spelling Correction**  
https://github.com/msalhab96/AraSpell

Original authors:

- Mahmoud Salhab
- Faisal Abu-Khzam

The original AraSpell source code is licensed under the MIT License. Where applicable, original copyright and license notices are retained for substantial portions of reused source code.

---

## Citation

If you use the research behind AraSpellX, please cite the original paper:

```bibtex
@article{salhab2024araspell,
  title={AraSpell: A Deep Learning Approach for Arabic Spelling Correction},
  author={Salhab, Mahmoud and Abu-Khzam, Faisal},
  journal={arXiv preprint arXiv:2405.06981},
  year={2024}
}
```

---

## License

AraSpellX is released under the **MIT License**, subject to the attribution requirements described in [`LICENSE`](LICENSE).

---

<p align="center">
  <sub>Arabic NLP • Spelling Correction • Seq2Seq • Transformer</sub>
</p>