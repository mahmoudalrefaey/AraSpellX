"""Deterministic held-out pages: reserved for test sets, never used for training.

A page's bucket is a hash of "wiki:page_id" modulo 10,000, so the split does
not depend on file order and anyone can recompute it from the page id:

- buckets [0, 50):    clean-text damage sets (T-7)
- buckets [50, 300):  Arabic Wikipedia pages whose real edits form T-1
- everything else:    available for training

Training excludes every page below TRAINING_FROM, so mining edits for T-1
later can never leak into training text that was built earlier.
"""
from __future__ import annotations

import hashlib

HELDOUT_PER_10000 = 50   # T-7 clean text (0.5% of pages)
EDIT_TEST_UNTIL = 300    # T-1 real edits come from buckets [50, 300) (2.5%)
TRAINING_FROM = EDIT_TEST_UNTIL


def bucket(wiki: str, page_id: int) -> int:
    digest = hashlib.blake2b(f"{wiki}:{page_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % 10000


def is_heldout(wiki: str, page_id: int) -> bool:
    """Page reserved for the clean-text test sets (T-7)."""
    return bucket(wiki, page_id) < HELDOUT_PER_10000


def is_edit_test(wiki: str, page_id: int) -> bool:
    """Page whose real edits belong to the typed-error test set (T-1)."""
    return HELDOUT_PER_10000 <= bucket(wiki, page_id) < EDIT_TEST_UNTIL


def is_training(wiki: str, page_id: int) -> bool:
    return bucket(wiki, page_id) >= TRAINING_FROM
