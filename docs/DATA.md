# Getting the data

v0 trains on the AraSpell corpus: sentences from the Arabic Wikipedia 2021 dump, each paired with synthetically distorted versions. The files are not in this repository (about 3 GB for the training set).

## Option 1: download the prepared CSV files
The AraSpell authors publish the prepared files on Google Drive (links from the [AraSpell README](https://github.com/msalhab96/AraSpell)):

| Split | Google Drive file id |
|---|---|
| Train | `1uu_Ga7MZ6sYHhxfuPBH3rVPglxmIqL0O` |
| Test | `1YwrWfISPXHQDaTtf3h6K-1-8bO3kH5lB` |
| Dev | `18As7vbgveFWsjt6wGqlgvjw8ax9FxJbF` |

```bash
pip install gdown
mkdir -p data/dataset
gdown 1uu_Ga7MZ6sYHhxfuPBH3rVPglxmIqL0O -O data/dataset/train.csv
gdown 1YwrWfISPXHQDaTtf3h6K-1-8bO3kH5lB -O data/dataset/test.csv
gdown 18As7vbgveFWsjt6wGqlgvjw8ax9FxJbF -O data/dataset/dev.csv
```

The copies used for the results in the README had the columns:

```text
,clean,distorted_0.05,distorted_0.1,distorted_0.15
```

with 6,922,318 training rows and 100,000 test rows. Check the header after downloading. The training scripts read the column given by `--dist_key` (default `distorted_<distortion_ratio>`); if your file has a single `distorted` column, pass `--dist_key distorted`.

On the first run the dataset is tokenized and cached next to the CSV (`*.tokenized.pt`, about 1.7 GB for the training set). The cache is rebuilt automatically when the CSV or tokenizer changes.

## Option 2: build the corpus from a Wikipedia dump
`data/process_data.py` is the original preprocessing pipeline:
- it splits text into lines on punctuation and removes diacritics, numbers and non-Arabic characters
- it keeps lines of 5–20 words and 15–128 characters
- it writes a CSV with distorted versions at several ratios

The source dump used by AraSpell is on [Kaggle](https://www.kaggle.com/datasets/z3rocool/arabic-wikipedia-dump-2021). The script expects plain-text files in `--data_path` and a JSON list of excluded words in `--execlude_words_files`; the exact preprocessing settings of the published files are not documented, so option 1 is recommended.

## Adding spelling habits
`data/regenerate_distortions.py` builds a `distorted_rw` column from the `clean` column: spelling habits plus typos at a per-sentence ratio. It writes `<input>_rw.csv`:

```bash
python -m data.regenerate_distortions --input data/dataset/test.csv --preview 20   # inspect samples
python -m data.regenerate_distortions --input data/dataset/train.csv
python -m data.regenerate_distortions --input data/dataset/test.csv
```

## License
The text comes from Wikipedia (CC BY-SA); keep attribution when redistributing it.
