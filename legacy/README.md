# Legacy: v0 seq2seq baseline

This folder holds v0 of AraSpellX: a reimplementation of the AraSpell sequence-to-sequence approach (character-level encoder-decoder trained on Arabic Wikipedia sentences with synthetic errors). It is no longer developed and is kept as the baseline for v1.

| Path | Content |
|---|---|
| `core/`, `models/`, `data/` | Training loop, encoder-decoder model, dataset, tokenizer and error generator |
| `train.py`, `evaluate.py` | Training and test-set evaluation |
| `inference.py` | Correction of raw text (padding, normalization, chunking) and `--eval_csv` scoring |
| `evaluate_probes.py` | Controlled typo and habit probes for checkpoint comparison |
| `docs/PERFORMANCE.md` | Training performance notes |

Run from the repository root, for example:

```bash
python legacy/train.py --train_path data/dataset/train_rw.csv --test_path data/dataset/test_rw.csv --dist_key distorted_rw
python legacy/inference.py --checkpoint outdir/<checkpoint>.pt --eval_csv benchmarks/real_world.csv --input_col distorted --ref_col clean
PYTHONPATH=legacy python -m data.regenerate_distortions --input data/dataset/test.csv --preview 20
```

Known limits that motivated v1: 40-token vocabulary (no punctuation, digits or Latin), whole-sentence rewriting without confidence, training only on rule-generated errors, and evaluation only on synthetic data.
