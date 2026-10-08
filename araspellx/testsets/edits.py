"""T-1: real typed errors mined from Arabic Wikipedia's edit history.

Consecutive revisions of every article are compared; a paragraph becomes a
pair (before, after) when the only differences are 1-5 small spelling fixes:
single words or word pairs (merges, splits) that change by at most two
characters, Arabic letters only. Fixes that a later revision undid are
dropped. Pages in the T-1 buckets (data/splits.py) give the test set; pages
available for training give real typed-error pairs for correction training.

The history dumps are 54 7-Zip files (11 GB) that expand to hundreds of GB,
so they are streamed through 7z and never written out. Run in the OCR
container, which has 7-Zip:

    <container> python3 -m araspellx.testsets.edits download --workers 2
    <container> python3 -m araspellx.testsets.edits mine --workers 6
    python -m araspellx.testsets.edits build
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import random
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from rapidfuzz.distance import Levenshtein

from araspellx.data.splits import is_edit_test, is_training
from araspellx.data.wikimedia import wikitext_to_paragraphs
from araspellx.text.charset import ARABIC
from araspellx.text.normalize import normalize

WIKI, DATE = "arwiki", "20261001"
DUMPS = f"https://dumps.wikimedia.org/{WIKI}/{DATE}/"
HISTORY = Path("data/raw/wikimedia/history")
# Wikimedia rejects requests without a descriptive user agent and asks for at most two parallel downloads.
HEADERS = {"User-Agent": "AraSpellX/1.0 (https://github.com/mahmoudalrefaey/AraSpellX)"}
PARTS = Path("data/v1/edits")
TEST_OUT = Path("data/v1/testsets/T1_wiki_edits.jsonl")
TRAIN_OUT = Path("data/v1/pairs/wiki_edits_train.jsonl")

MAX_BLOCK = 1500      # largest changed region between two revisions (characters)
MAX_LINES = 5         # largest number of changed lines
MAX_CHANGES = 5       # spelling fixes per paragraph
MAX_DISTANCE = 2      # characters changed per fix
MAX_SAME_FIX = {"test": 3, "train": 30}  # repeats of one (before, after) fix, e.g. from a bot rule


# --- download -------------------------------------------------------------------------------

def dump_files() -> Dict[str, Dict]:
    with urllib.request.urlopen(urllib.request.Request(DUMPS + "dumpstatus.json", headers=HEADERS)) as response:
        status = json.load(response)
    return status["jobs"]["metahistory7zdump"]["files"]


def _sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(job: Tuple[str, Dict]) -> str:
    name, info = job
    target = HISTORY / name
    if target.exists() and target.stat().st_size == info["size"]:
        return f"{name}: already there"
    partial = target.with_suffix(target.suffix + ".partial")
    done = partial.stat().st_size if partial.exists() else 0
    request = urllib.request.Request(DUMPS + name, headers={**HEADERS, **({"Range": f"bytes={done}-"} if done else {})})
    with urllib.request.urlopen(request) as response, open(partial, "ab" if done else "wb") as f:
        while block := response.read(1 << 20):
            f.write(block)
    if _sha1(partial) != info["sha1"]:
        partial.unlink()
        return f"{name}: checksum mismatch, deleted (run download again)"
    partial.replace(target)
    return f"{name}: {info['size'] / 1e6:.0f} MB"


def download(workers: int) -> None:
    HISTORY.mkdir(parents=True, exist_ok=True)
    files = sorted(dump_files().items())
    with Pool(workers) as pool:
        for i, message in enumerate(pool.imap_unordered(_download, files), 1):
            print(f"[{i}/{len(files)}] {message}", flush=True)


# --- mining ---------------------------------------------------------------------------------

def _common_prefix(a: str, b: str) -> int:
    low, high = 0, min(len(a), len(b))
    while low < high:  # largest k with a[:k] == b[:k], by bisection on C-level comparisons
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def _common_suffix(a: str, b: str, limit: int) -> int:
    low, high = 0, limit
    while low < high:
        middle = (low + high + 1) // 2
        if a[len(a) - middle:] == b[len(b) - middle:]:
            low = middle
        else:
            high = middle - 1
    return low


def changed_lines(before: str, after: str) -> List[Tuple[str, str]]:
    """Pairs of lines that differ when one small region of the text changed; [] otherwise."""
    if before == after:
        return []
    prefix = _common_prefix(before, after)
    suffix = _common_suffix(before, after, min(len(before), len(after)) - prefix)
    if max(len(before), len(after)) - prefix - suffix > MAX_BLOCK:
        return []
    start = before.rfind("\n", 0, prefix) + 1
    end = before.find("\n", len(before) - suffix)
    end = len(before) if end < 0 else end
    old = before[start:end].split("\n")
    new = after[start:end + len(after) - len(before)].split("\n")
    if len(old) != len(new) or len(old) > MAX_LINES:
        return []
    return [(o, n) for o, n in zip(old, new) if o != n]


PREFIXES = ("ال", "و", "ف", "ب", "ك", "ل")  # adding or removing one is grammar, not spelling


def _arabic_word(text: str) -> bool:
    return bool(text) and all(c in ARABIC or c == " " for c in text) and sum(c in ARABIC for c in text) >= 2


def _prefix_only(source: str, target: str) -> bool:
    return any(target == p + source or source == p + target for p in PREFIXES)


def spelling_fixes(before: str, after: str) -> Optional[List[Tuple[str, str]]]:
    """The (before, after) word fixes between two plain-text paragraphs, or None if they differ otherwise."""
    old, new = before.split(), after.split()
    fixes = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        if tag != "replace":
            return None  # words added or removed
        if i2 - i1 == j2 - j1:  # neighbouring words, each fixed on its own
            pairs = [(o, n) for o, n in zip(old[i1:i2], new[j1:j2]) if o != n]
        elif sorted((i2 - i1, j2 - j1)) == [1, 2]:  # two words merged, or one split
            pairs = [(" ".join(old[i1:i2]), " ".join(new[j1:j2]))]
        else:
            return None  # a longer rewrite
        for source, target in pairs:
            if (not (_arabic_word(source) and _arabic_word(target)) or _prefix_only(source, target)
                    or Levenshtein.distance(source, target) > MAX_DISTANCE):
                return None
            fixes.append((source, target))
    return fixes if 1 <= len(fixes) <= MAX_CHANGES else None


def paragraph_pair(old_line: str, new_line: str) -> Optional[Dict]:
    old, new = wikitext_to_paragraphs(old_line), wikitext_to_paragraphs(new_line)
    if len(old) != 1 or len(new) != 1:
        return None
    before, after = normalize(old[0]).text, normalize(new[0]).text
    fixes = spelling_fixes(before, after)
    return None if fixes is None else {"noisy": before, "clean": after, "fixes": fixes}


def _pages(stream) -> Iterator[Tuple[int, Iterator[Dict]]]:
    """(page id, revisions) of main-namespace pages, streamed; revisions must be consumed in order."""
    context = ET.iterparse(stream, events=("start", "end"))
    _, root = next(context)
    page_id, ns, in_revision = None, None, False

    def revisions():
        nonlocal in_revision
        for event, elem in context:
            tag = elem.tag.rsplit("}", 1)[-1]
            if event == "start":
                in_revision = in_revision or tag == "revision"
                continue
            if tag == "revision":
                in_revision = False
                yield {"id": elem.findtext("{*}id"), "comment": elem.findtext("{*}comment") or "",
                       "user": elem.findtext("{*}contributor/{*}username") or "",
                       "text": elem.findtext("{*}text") or ""}
                elem.clear()
            elif tag == "page":
                root.clear()
                return

    for event, elem in context:
        tag = elem.tag.rsplit("}", 1)[-1]
        if event == "start" and tag == "page":
            page_id, ns = None, None
        elif event == "end" and tag == "ns":
            ns = elem.text
        elif event == "end" and tag == "id" and page_id is None:
            page_id = int(elem.text)
        elif event == "start" and tag == "revision":
            in_revision = True
            if ns == "0" and page_id is not None:
                yield page_id, revisions()
            else:
                for _ in revisions():
                    pass


def mine_file(path: Path) -> str:
    target = PARTS / (path.name + ".jsonl")
    if target.exists():
        return f"{path.name}: already mined"
    process = subprocess.Popen(["7z", "e", "-so", str(path)], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    counts = Counter()
    partial = target.with_suffix(".partial")
    with open(partial, "w", encoding="utf-8") as out:
        for page_id, revisions in _pages(process.stdout):
            split = "test" if is_edit_test(WIKI, page_id) else "train" if is_training(WIKI, page_id) else None
            previous, found = "", []
            for revision in revisions:
                text = revision["text"]
                if split and previous and abs(len(text) - len(previous)) <= 200:
                    for old_line, new_line in changed_lines(previous, text):
                        pair = paragraph_pair(old_line, new_line)
                        if pair:
                            found.append({"id": f"{WIKI}:{page_id}:{revision['id']}", "split": split,
                                          "comment": revision["comment"], "user": revision["user"], **pair})
                previous = text
            counts["pages"] += 1
            if not found:
                continue
            final = normalize(previous).text  # keep fixes that the latest revision still has
            for pair in found:
                if all(after in final for _, after in pair["fixes"]):
                    out.write(json.dumps(pair, ensure_ascii=False) + "\n")
                    counts[pair["split"]] += 1
                else:
                    counts["undone"] += 1
    process.wait()
    partial.replace(target)
    return f"{path.name}: {dict(counts)}"


def mine(workers: int) -> None:
    PARTS.mkdir(parents=True, exist_ok=True)
    files = sorted(HISTORY.glob("*.7z"))
    with Pool(workers) as pool:
        for i, message in enumerate(pool.imap_unordered(mine_file, files), 1):
            print(f"[{i}/{len(files)}] {message}", flush=True)


# --- build ----------------------------------------------------------------------------------

def build(seed: int = 0) -> None:
    rows = []
    for part in sorted(PARTS.glob("*.jsonl")):
        with open(part, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f]
    random.Random(seed).shuffle(rows)
    seen, per_fix, kept = set(), Counter(), {"test": [], "train": []}
    for row in rows:
        key = hashlib.blake2b(f"{row['noisy']}\n{row['clean']}".encode(), digest_size=8).digest()
        fixes = [tuple(f) for f in row["fixes"]]
        if key in seen or any(per_fix[(row["split"], f)] >= MAX_SAME_FIX[row["split"]] for f in fixes):
            continue
        seen.add(key)
        per_fix.update((row["split"], f) for f in fixes)
        kept[row["split"]].append(row)
    for split, path in (("test", TEST_OUT), ("train", TRAIN_OUT)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for row in kept[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        fixes = Counter(f"{a} -> {b}" for row in kept[split] for a, b in row["fixes"])
        print(f"{split}: {len(kept[split])} paragraphs, {sum(fixes.values())} fixes -> {path}")
        print("  most common:", fixes.most_common(8))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "mine", "build"])
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    if args.command == "download":
        download(args.workers)
    elif args.command == "mine":
        mine(args.workers)
    else:
        build()


if __name__ == "__main__":
    sys.exit(main())
