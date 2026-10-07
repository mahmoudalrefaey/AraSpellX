"""Real OCR pairs from the Yarmouk Arabic OCR Dataset (T-4, T-5 exclusions, training).

Yarmouk has real 300-dpi scans of 4,587 printed Arabic Wikipedia articles,
their original HTML (ground truth) and the authors' ABBYY OCR output.

`ocr` (run in the OCR container) OCRs every scan with the project's fixed
Tesseract and caches the text per article. `build` aligns each article's
OCR text with its ground-truth paragraphs and writes (noisy, clean)
paragraph pairs for every split:

- test:  300 articles of Yarmouk's Testing split (T-4)
- dev:   300 other Testing articles (threshold and calibration tuning)
- train: all other articles, except the 100 pages reused by NOD (T-5)

    docker run --rm -v "$PWD:/work" -w /work araspellx-ocr python3 -m araspellx.testsets.yarmouk ocr
    python -m araspellx.testsets.yarmouk build
"""
from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import subprocess
import tempfile
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List, Tuple

from rapidfuzz.distance import Levenshtein

from araspellx.eval.metrics import boundary_map

ROOT = Path("data/raw/yarmouk")
SPLITS = {"Training": ROOT / "Training/Training", "Testing": ROOT / "Testing/Testing"}
PDF_DIRS = {"Training": "Scanned/training/training", "Testing": "Scanned/testing/testing"}
NOD_GT = Path("data/raw/nod/gt/ground_truth/yarmouk_gt")
TESSERACT_CACHE = Path("data/v1/ocr/yarmouk_tesseract")
OUT = Path("data/v1/testsets")
TRAIN_OUT = Path("data/v1/pairs")

PARAGRAPH = re.compile(r"(?is)<(h1|p|li|h[2-6])\b[^>]*>(.*?)</\1>")
TAGS = re.compile(r"<[^>]+>")
SPACES = re.compile(r"\s+")
MAX_PAIR_CER = 0.5  # pairs above this are alignment failures, not OCR errors


def article_ids(split: str) -> List[str]:
    return sorted(f.split(".")[0] for f in os.listdir(SPLITS[split] / "HTML/html/html"))


def ground_truth(split: str, article: str) -> List[Tuple[str, bool]]:
    """The article's text blocks in print order, each flagged as a paragraph or not.

    The title (h1) is printed on the scan, so it takes part in the alignment,
    but it is not a paragraph and never becomes a pair.
    """
    raw = (SPLITS[split] / "HTML/html/html" / f"{article}.htm").read_bytes().decode("utf-8", "replace")
    blocks = []
    for tag, body in PARAGRAPH.findall(raw):
        text = SPACES.sub(" ", html.unescape(TAGS.sub(" ", body))).strip()
        if tag.lower() == "h1" or len(text) >= 20:
            blocks.append((text, tag.lower() != "h1"))
    return blocks


def abbyy_text(split: str, article: str) -> str:
    path = SPLITS[split] / "OCR/text/text" / f"{article}.txt"
    return SPACES.sub(" ", path.read_bytes().decode("utf-8", "replace")).strip()


def _tesseract_article(job: Tuple[str, str]) -> str:
    split, article = job
    target = TESSERACT_CACHE / f"{article}.txt"
    if target.exists():
        return article
    pdf = SPLITS[split] / PDF_DIRS[split] / f"{article}.pdf"
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", "300", "-gray", "-png", str(pdf), f"{tmp}/page"],
                       check=True, capture_output=True)
        texts = []
        for page in sorted(Path(tmp).glob("page*.png")):
            result = subprocess.run(["tesseract", str(page), "stdout", "-l", "ara", "--psm", "6"],
                                    check=True, capture_output=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
            texts.append(result.stdout.decode("utf-8"))
    partial = target.with_suffix(".partial")  # renamed when complete, so stopping is safe
    partial.write_text(SPACES.sub(" ", " ".join(texts)).strip(), encoding="utf-8")
    partial.replace(target)
    return article


def run_ocr(workers: int) -> None:
    TESSERACT_CACHE.mkdir(parents=True, exist_ok=True)
    # Test and dev articles first, so evaluation sets are ready before training pairs.
    splits = split_articles()
    jobs = splits["test"] + splits["dev"] + splits["train"]
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_tesseract_article, jobs), 1):
            if i % 200 == 0:
                print(f"{i}/{len(jobs)} articles", flush=True)


def paragraph_pairs(blocks: List[Tuple[str, bool]], ocr_text: str) -> List[Tuple[str, str]]:
    """Cut the article's OCR text at the ground-truth block boundaries."""
    truth = " ".join(text for text, _ in blocks)
    mapping = boundary_map(truth, ocr_text)
    pairs, start = [], 0
    for text, is_paragraph in blocks:
        end = start + len(text)
        noisy = ocr_text[mapping[start]:mapping[end]].strip()
        if is_paragraph and noisy and Levenshtein.normalized_distance(noisy, text) <= MAX_PAIR_CER:
            pairs.append((noisy, text))
        start = end + 1
    return pairs


def split_articles(seed: int = 0) -> Dict[str, List[Tuple[str, str]]]:
    nod = {f.split(".")[0] for f in os.listdir(NOD_GT)}
    testing = article_ids("Testing")
    random.Random(seed).shuffle(testing)
    return {
        "test": [("Testing", a) for a in sorted(testing[:300])],
        "dev": [("Testing", a) for a in sorted(testing[300:600])],
        "train": [("Testing", a) for a in sorted(testing[600:])]
                 + [("Training", a) for a in article_ids("Training") if a not in nod],
        "nod": [("Training", a) for a in sorted(nod)],
    }


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    TRAIN_OUT.mkdir(parents=True, exist_ok=True)
    targets = {"test": OUT / "T4_yarmouk_test.jsonl", "dev": OUT / "yarmouk_dev.jsonl",
               "train": TRAIN_OUT / "yarmouk_train.jsonl"}
    for split, articles in split_articles().items():
        if split == "nod":
            continue
        counts = {"tesseract": 0, "abbyy": 0}
        with open(targets[split], "w", encoding="utf-8") as f:
            for source_split, article in articles:
                blocks = ground_truth(source_split, article)
                engines = {"abbyy": abbyy_text(source_split, article)}
                cached = TESSERACT_CACHE / f"{article}.txt"
                if cached.exists():
                    engines["tesseract"] = cached.read_text(encoding="utf-8")
                for engine, text in engines.items():
                    for i, (noisy, clean) in enumerate(paragraph_pairs(blocks, text)):
                        f.write(json.dumps({"id": f"yarmouk:{article}:{i}", "engine": engine,
                                            "noisy": noisy, "clean": clean}, ensure_ascii=False) + "\n")
                        counts[engine] += 1
        print(f"{split}: {len(articles)} articles, pairs {counts} -> {targets[split]}")
    with open(OUT / "nod_articles.json", "w", encoding="utf-8") as f:
        json.dump([a for _, a in split_articles()["nod"]], f)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["ocr", "build"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run_ocr(args.workers) if args.command == "ocr" else build()


if __name__ == "__main__":
    main()
