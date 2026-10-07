"""Deterministic held-out pages: reserved for test sets, never used for training.

A page is held out when a hash of "wiki:page_id" falls in the first
HELDOUT_PER_10000 of 10,000 buckets, so the split does not depend on file
order and can be recomputed by anyone from the page id alone.
"""
from __future__ import annotations

import hashlib

HELDOUT_PER_10000 = 50  # 0.5% of pages


def bucket(wiki: str, page_id: int) -> int:
    digest = hashlib.blake2b(f"{wiki}:{page_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000


def is_heldout(wiki: str, page_id: int) -> bool:
    return bucket(wiki, page_id) < HELDOUT_PER_10000
