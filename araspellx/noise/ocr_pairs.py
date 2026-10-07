"""Make (OCR output, clean text) training pairs by rendering, degrading and OCR'ing.

Runs in the OCR container. Paragraphs come from training pages of Arabic
Wikipedia (modern) and Arabic Wikisource (classical), skipping any that
overlap a test set. Each paragraph is rendered in a random open font and
size, degraded (ocr/degrade.py) and read back by the fixed Tesseract, so
the errors are those of a real OCR engine on imperfect pages.

    docker run --rm -v "$PWD:/work" -w /work araspellx-ocr python3 -m araspellx.noise.ocr_pairs --count 100000
"""
from __future__ import annotations

import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, List

from rapidfuzz.distance import Levenshtein

from araspellx.data.dedup import overlaps, test_shingles
from araspellx.data.splits import is_training
from araspellx.ocr.degrade import degrade
from araspellx.ocr.render import FONTS, render
from araspellx.ocr.tesseract import ocr
from araspellx.text.normalize import normalize

SOURCES = {"arwiki": 0.7, "arwikisource": 0.3}  # share of paragraphs per source
MIN_CHARS, MAX_CHARS = 80, 600
MAX_PAIR_CER = 0.5  # pages the OCR could not read at all teach nothing


def sample_paragraphs(corpus: Path, count: int, banned: set, seed: int) -> List[Dict]:
    """Reservoir-sample training paragraphs of a useful length from each source."""
    rng = random.Random(seed)
    chosen = []
    for wiki, share in SOURCES.items():
        want, seen, reservoir = int(count * share), 0, []
        with open(corpus / f"{wiki}.jsonl", encoding="utf-8") as f:
            for line in f:
                page = json.loads(line)
                if not is_training(wiki, page["page_id"]):
                    continue
                for i, text in enumerate(page["paragraphs"]):
                    if not MIN_CHARS <= len(text) <= MAX_CHARS:
                        continue
                    seen += 1
                    item = {"id": f"{wiki}:{page['page_id']}:{i}", "text": text}
                    if len(reservoir) < want:
                        reservoir.append(item)
                    elif rng.random() < want / seen:
                        reservoir[rng.randrange(want)] = item
        chosen += [p for p in reservoir if not overlaps(p["text"], banned)]
    rng.shuffle(chosen)
    return chosen


def _make_pair(job: Dict) -> Dict:
    rng = random.Random(job["seed"])
    clean = normalize(job["text"]).text
    font = rng.choice(sorted(FONTS))
    page = render(clean, font=font, points=rng.choice([10, 11, 12, 14, 16]), dpi=300)
    image, params = degrade(page.image, rng)
    noisy = ocr(image)
    return {"id": job["id"], "noisy": noisy, "clean": clean, "font": font, "degrade": params,
            "cer": round(Levenshtein.normalized_distance(noisy, clean), 4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/v1/corpus"))
    parser.add_argument("--tests", type=Path, nargs="+",
                        default=[Path("data/v1/testsets"), Path("data/raw/nod/gt/ground_truth/yarmouk_gt"),
                                 Path("data/raw/openiti/ara")])
    parser.add_argument("--out", type=Path, default=Path("data/v1/pairs/ocr_render.jsonl"))
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    done = set()
    if args.out.exists():  # resume: skip paragraphs already rendered
        with open(args.out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    paragraphs = sample_paragraphs(args.corpus, args.count, test_shingles(args.tests), args.seed)
    jobs = [{"id": p["id"], "text": p["text"], "seed": hash((args.seed, p["id"])) & 0xFFFFFFF}
            for p in paragraphs if p["id"] not in done]
    print(f"{len(paragraphs)} paragraphs sampled, {len(jobs)} to render", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a", encoding="utf-8") as out, Pool(args.workers) as pool:
        for i, pair in enumerate(pool.imap_unordered(_make_pair, jobs, chunksize=4), 1):
            out.write(json.dumps(pair, ensure_ascii=False) + "\n")
            if i % 1000 == 0:
                out.flush()
                print(f"{i}/{len(jobs)} pairs", flush=True)


if __name__ == "__main__":
    main()
