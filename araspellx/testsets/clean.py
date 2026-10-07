"""Build T-7: clean text for measuring damage (gate G1).

Each set holds about 50k words of correct text from held-out pages that
are never used for training: modern standard Arabic (Arabic Wikipedia),
classical Arabic (Arabic Wikisource), diacritized text (paragraphs where
most letters carry harakat) and dialect (Moroccan and Egyptian
Wikipedia paragraphs with clear dialect markers). A model may change
these texts only by mistake, so every change counts as damage.

    python -m araspellx.testsets.clean --corpus data/v1/corpus --out data/v1/testsets
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Callable, Dict, Iterator, List

from araspellx.data.splits import is_heldout
from araspellx.text.charset import ARABIC, DIACRITICS

TARGET_WORDS = 50_000

DIALECT_MARKERS = re.compile(
    r"(?:^|\s)(?:"
    # Moroccan
    r"ديال|ديالو|ديالها|كاين|كاينة|كاينين|بزاف|واش|دابا|هادشي|هادي|هاد|ماشي|كيكون|كاتكون|غادي|لي|ف|ؤ"
    r"|"
    # Egyptian
    r"ده|دى|دي|اللى|مش|عشان|بتاع|بتاعة|كده|ازاى|دلوقتى|اوى|بيقول|هيكون|برضه|علشان|بقى|حاجة"
    r")(?=\s|$|[،.:])")


def diacritic_share(text: str) -> float:
    letters = sum(c in ARABIC for c in text)
    return sum(c in DIACRITICS for c in text) / max(letters, 1)


def heldout_paragraphs(path: Path, wiki: str) -> Iterator[Dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            page = json.loads(line)
            if is_heldout(wiki, page["page_id"]):
                for i, text in enumerate(page["paragraphs"]):
                    yield {"id": f"{wiki}:{page['page_id']}:{i}", "text": text,
                           "wiki": wiki, "page_id": page["page_id"], "title": page["title"]}


def collect(sources: List[Iterator[Dict]], keep: Callable[[str], bool], seed: int) -> List[Dict]:
    """Shuffle the matching held-out paragraphs and take ~TARGET_WORDS words."""
    candidates = [p for source in sources for p in source if keep(p["text"])]
    random.Random(seed).shuffle(candidates)
    chosen, words = [], 0
    for paragraph in candidates:
        if words >= TARGET_WORDS:
            break
        chosen.append(paragraph)
        words += len(paragraph["text"].split())
    return chosen


def build(corpus: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    plain = lambda t: diacritic_share(t) < 0.05
    sets = {
        "T7_msa": ([heldout_paragraphs(corpus / "arwiki.jsonl", "arwiki")],
                   lambda t: plain(t) and len(DIALECT_MARKERS.findall(t)) == 0),
        "T7_classical": ([heldout_paragraphs(corpus / "arwikisource.jsonl", "arwikisource")], plain),
        "T7_diacritized": ([heldout_paragraphs(corpus / "arwikisource.jsonl", "arwikisource"),
                            heldout_paragraphs(corpus / "arwiki.jsonl", "arwiki")],
                           lambda t: diacritic_share(t) >= 0.5),
        "T7_dialect": ([heldout_paragraphs(corpus / "arywiki.jsonl", "arywiki"),
                        heldout_paragraphs(corpus / "arzwiki.jsonl", "arzwiki")],
                       lambda t: plain(t) and len(DIALECT_MARKERS.findall(t)) >= 3),
    }
    for seed, (name, (sources, keep)) in enumerate(sets.items()):
        chosen = collect(sources, keep, seed)
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for paragraph in chosen:
                f.write(json.dumps(paragraph, ensure_ascii=False) + "\n")
        words = sum(len(p["text"].split()) for p in chosen)
        print(f"{name}: {len(chosen)} paragraphs, {words} words")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", type=Path, default=Path("data/v1/corpus"))
    parser.add_argument("--out", type=Path, default=Path("data/v1/testsets"))
    args = parser.parse_args()
    build(args.corpus, args.out)


if __name__ == "__main__":
    main()
