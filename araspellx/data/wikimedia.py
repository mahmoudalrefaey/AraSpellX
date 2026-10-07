"""Turn Wikimedia XML dumps into clean paragraphs of plain text.

Unlike the AraSpell corpus (single sentences without punctuation or
digits), paragraphs keep punctuation, numbers and Latin words, as real
pipeline text does. Markup, templates, references, tables, files and
categories are removed; link text is kept.

    python -m araspellx.data.wikimedia data/raw/wikimedia/arywiki-20261001-pages-articles.xml.bz2 \
        data/v1/corpus/arywiki.jsonl
"""
from __future__ import annotations

import argparse
import bz2
import json
import re
import sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterator, List, Optional

import mwparserfromhell

from araspellx.text.charset import ARABIC

# Namespace prefixes of links that are not article text (files and categories).
NON_TEXT_LINKS = re.compile(
    r"^\s*:?\s*(file|image|category|media|ملف|صورة|تصنيف|وسائط|فئة)\s*:", re.IGNORECASE)
REMOVED_TAGS = re.compile(
    r"<(ref|gallery|math|chem|score|timeline|imagemap|syntaxhighlight|source|pre|code|graph|mapframe)"
    r"\b[^>]*?(/>|>.*?</\1\s*>)", re.IGNORECASE | re.DOTALL)
COMMENTS = re.compile(r"<!--.*?-->", re.DOTALL)
HTML_TAGS = re.compile(r"</?[a-zA-Z][^>]*>")
TABLE = re.compile(r"\{\|(?:(?!\{\|).)*?\|\}", re.DOTALL)
LIST_MARKERS = re.compile(r"^[*#:;]+\s*")
SPACES = re.compile(r"[ \t ]+")
LEFTOVER_MARKUP = re.compile(r"\[\[|\]\]|\{\{|\}\}|\|\||^\s*[|!]|''")

MIN_CHARS = 40
MIN_ARABIC_SHARE = 0.6


def _remove_tables(text: str) -> str:
    previous = None
    while previous != text:  # innermost tables first, until none are left
        previous, text = text, TABLE.sub("", text)
    return text


def wikitext_to_paragraphs(wikitext: str) -> List[str]:
    text = COMMENTS.sub("", wikitext)
    text = REMOVED_TAGS.sub("", text)
    text = _remove_tables(text)
    code = mwparserfromhell.parse(text)
    for template in code.filter_templates(recursive=False):
        try:
            code.remove(template)
        except ValueError:
            pass
    for link in code.filter_wikilinks():
        if NON_TEXT_LINKS.match(str(link.title)):
            try:
                code.remove(link)
            except ValueError:
                pass
    plain = code.strip_code(normalize=True, collapse=True)
    plain = HTML_TAGS.sub("", plain)

    paragraphs = []
    for line in plain.split("\n"):
        line = line.strip()
        if not line or line.startswith("=") or LEFTOVER_MARKUP.search(line):
            continue
        line = SPACES.sub(" ", LIST_MARKERS.sub("", line)).strip()
        letters = [c for c in line if c.isalpha()]
        if len(line) < MIN_CHARS or not letters:
            continue
        if sum(c in ARABIC for c in letters) / len(letters) < MIN_ARABIC_SHARE:
            continue
        paragraphs.append(line)
    return paragraphs


def iter_pages(dump: Path) -> Iterator[Dict[str, str]]:
    """Yield main-namespace, non-redirect pages from a pages-articles dump."""
    with bz2.open(dump, "rb") as f:
        context = ET.iterparse(f, events=("end",))
        for _, elem in context:
            if not elem.tag.endswith("}page"):
                continue
            ns = elem.findtext("{*}ns")
            redirect = elem.find("{*}redirect")
            if ns == "0" and redirect is None:
                yield {
                    "page_id": elem.findtext("{*}id"),
                    "title": elem.findtext("{*}title"),
                    "text": elem.findtext("{*}revision/{*}text") or "",
                }
            elem.clear()


def _process(page: Dict[str, str]) -> Optional[str]:
    paragraphs = wikitext_to_paragraphs(page["text"])
    if not paragraphs:
        return None
    return json.dumps({"page_id": int(page["page_id"]), "title": page["title"],
                       "paragraphs": paragraphs}, ensure_ascii=False)


def extract(dump: Path, output: Path, workers: int = 8) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    pages = kept = 0
    with open(output, "w", encoding="utf-8") as out, Pool(workers) as pool:
        for line in pool.imap(_process, iter_pages(dump), chunksize=64):
            pages += 1
            if line is not None:
                out.write(line + "\n")
                kept += 1
            if pages % 50000 == 0:
                print(f"{pages} pages, {kept} with text", file=sys.stderr)
    print(f"{dump.name}: {pages} pages, {kept} with text -> {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dump", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    extract(args.dump, args.output, args.workers)


if __name__ == "__main__":
    main()
