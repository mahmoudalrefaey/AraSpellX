"""T-6: real OCR of classical Arabic books (OpenITI gold standard; evaluation only).

OpenITI's gold standard has line images of printed classical books with
human double-checked transcriptions (CC BY-NC-SA 4.0: used for evaluation
only, never trained on or redistributed). Each line is OCR'd with the
fixed Tesseract; groups of consecutive lines become one test paragraph so
the model has context. OpenITI's own OCR output of each line is kept as a
second engine.

    docker run --rm -v "$PWD:/work" -w /work araspellx-ocr python3 -m araspellx.testsets.openiti ocr
    python -m araspellx.testsets.openiti build
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from multiprocessing import Pool
from pathlib import Path
from typing import List

ROOT = Path("data/raw/openiti/ara")
CACHE = Path("data/v1/ocr/openiti_tesseract")
OUT = Path("data/v1/testsets/T6_openiti.jsonl")
LINES_PER_PARAGRAPH = 5
SPACES = re.compile(r"\s+")


def gold_lines() -> List[Path]:
    return sorted(ROOT.glob("*/7_final/*.gt.txt"))


def _image(gt: Path) -> Path:
    return gt.with_name(gt.name[:-len(".gt.txt")] + ".png")


def _cache_path(gt: Path) -> Path:
    return CACHE / gt.parts[-3] / (gt.name[:-len(".gt.txt")] + ".txt")


def _ocr_line(gt: Path) -> None:
    target = _cache_path(gt)
    if target.exists():
        return
    result = subprocess.run(["tesseract", str(_image(gt)), "stdout", "-l", "ara", "--psm", "7"],
                            check=True, capture_output=True, env={**os.environ, "OMP_THREAD_LIMIT": "1"})
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".partial")
    partial.write_text(SPACES.sub(" ", result.stdout.decode("utf-8")).strip(), encoding="utf-8")
    partial.replace(target)


def run_ocr(workers: int) -> None:
    lines = gold_lines()
    with Pool(workers) as pool:
        pool.map(_ocr_line, lines, chunksize=32)
    print(f"{len(lines)} lines OCR'd")


def _read(path: Path) -> str:
    return SPACES.sub(" ", path.read_text(encoding="utf-8")).strip() if path.exists() else ""


def build() -> None:
    by_book = {}
    for gt in gold_lines():
        by_book.setdefault(gt.parts[-3], []).append(gt)
    count = 0
    with open(OUT, "w", encoding="utf-8") as f:
        for book, lines in sorted(by_book.items()):
            for start in range(0, len(lines), LINES_PER_PARAGRAPH):
                group = lines[start:start + LINES_PER_PARAGRAPH]
                clean = " ".join(_read(gt) for gt in group)
                engines = {
                    "tesseract": " ".join(_read(_cache_path(gt)) for gt in group),
                    "openiti": " ".join(_read(gt.with_name(gt.name[:-len(".gt.txt")] + ".png.rec.txt"))
                                        for gt in group),
                }
                for engine, noisy in engines.items():
                    if noisy and clean:
                        f.write(json.dumps({"id": f"openiti:{book}:{start}", "engine": engine,
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
