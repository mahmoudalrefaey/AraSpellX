"""Character tokenizer: one token per character, loadable with AutoTokenizer.

The released tokenizer is a standard PreTrainedTokenizerFast built with
the `tokenizers` library, so `AutoTokenizer.from_pretrained` works without
custom code. `encode_chars` is the fast path used inside AraSpellX; both
produce the same ids.
"""
from __future__ import annotations

from typing import List

from tokenizers import Regex, Tokenizer, models, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

from araspellx.text.charset import CLS, MASK, PAD, SEP, TOKEN_TO_ID, UNK

UNK_ID = TOKEN_TO_ID[UNK]


def build_tokenizer() -> PreTrainedTokenizerFast:
    tokenizer = Tokenizer(models.WordLevel(vocab=dict(TOKEN_TO_ID), unk_token=UNK))
    # Every character, newlines included, is its own pre-token.
    tokenizer.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    tokenizer.post_processor = processors.TemplateProcessing(
        single=f"{CLS} $A {SEP}",
        pair=f"{CLS} $A {SEP} $B {SEP}",
        special_tokens=[(CLS, TOKEN_TO_ID[CLS]), (SEP, TOKEN_TO_ID[SEP])],
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token=UNK, pad_token=PAD, cls_token=CLS, sep_token=SEP, mask_token=MASK,
        model_input_names=["input_ids", "attention_mask"],
        clean_up_tokenization_spaces=False,
    )


def encode_chars(text: str) -> List[int]:
    """Token ids of the characters of text, without [CLS]/[SEP]."""
    return [TOKEN_TO_ID.get(char, UNK_ID) for char in text]
