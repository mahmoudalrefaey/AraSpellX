"""T-5: degraded real scans from NOD (100 Yarmouk pages, each in many noise versions).

`ocr` (run in the OCR container) unpacks one noise version at a time into
the container's temporary space, OCRs every page with the fixed Tesseract,
keeps only the text and discards the images (they take ~1 GB per version).
`build` aligns each article's OCR text with NOD's ground truth.

    docker run --rm -v "$PWD:/work" -w /work araspellx-ocr python3 -m araspellx.testsets.nod ocr
    python -m araspellx.testsets.nod build
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

from araspellx.testsets.yarmouk import paragraph_pairs

ROOT = Path("data/raw/nod")
GT = ROOT / "gt/ground_truth/yarmouk_gt"
CACHE = Path("data/v1/ocr/nod")
OUT = Path("data/v1/testsets/T5_nod.jsonl")
SPACES = re.compile(r"\s+")


def _ocr_page(path: str) -> Tuple[str, str]:
    result = subprocess.run(["tesseract", path, "stdout", "-l", "ara", "--psm", "6"],
                            check=True, capture_output=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    return path, result.stdout.decode("utf-8")


def run_ocr(workers: int) -> None:
    for archive in sorted(ROOT.glob("yarmouk_*.tar.lzma")):
        variant = archive.name[len("yarmouk_"):-len(".tar.lzma")]
        target = CACHE / variant
        if target.exists() and len(list(target.glob("*.txt"))) >= 100:
            continue
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(["tar", "--lzma", "-xf", str(archive), "-C", tmp], check=True)
            pages = sorted(str(p) for p in Path(tmp).rglob("*.tif*"))
            with Pool(workers) as pool:
                texts = dict(pool.map(_ocr_page, pages))
        articles = defaultdict(list)
        for page in sorted(texts, key=lambda p: (Path(p).stem.split("_")[0], int(Path(p).stem.split("_")[1]))):
            articles[Path(page).stem.split("_")[0]].append(texts[page])
        for article, page_texts in articles.items():
            (target / f"{article}.txt").write_text(SPACES.sub(" ", " ".join(page_texts)).strip(),
                                                   encoding="utf-8")
        print(f"{variant}: {len(pages)} pages, {len(articles)} articles", flush=True)


def ground_truth(article: str) -> List[Tuple[str, bool]]:
    """NOD ground truth: first line is the title; paragraphs are separated by blank lines."""
    text = (GT / f"{article}.txt").read_text(encoding="utf-8")
    blocks = []
    for i, block in enumerate(re.split(r"\n\s*\n", text)):
        block = SPACES.sub(" ", block).strip()
        if block:
            arabic = sum(0x0600 <= ord(c) <= 0x06FF for c in block) / max(len(block), 1)
            blocks.append((block, i > 0 and len(block) >= 20 and arabic >= 0.5))
    return blocks


def build() -> None:
    count = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for variant_dir in sorted(CACHE.iterdir()):
            for cached in sorted(variant_dir.glob("*.txt")):
                article = cached.stem
                pairs = paragraph_pairs(ground_truth(article), cached.read_text(encoding="utf-8"))
                for i, (noisy, clean) in enumerate(pairs):
                    f.write(json.dumps({"id": f"nod:{variant_dir.name}:{article}:{i}",
                                        "variant": variant_dir.name, "engine": "tesseract",
                                        "noisy": noisy, "clean": clean}, ensure_ascii=False) + "\n")
                    count += 1
    print(f"{count} pairs -> {OUT}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["ocr", "build"])
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    run_ocr(args.workers) if args.command == "ocr" else build()


if __name__ == "__main__":
    main()
