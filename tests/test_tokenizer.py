from transformers import AutoTokenizer

from araspellx.text.charset import CLS, SEP, TOKEN_TO_ID, UNK, VOCAB
from araspellx.text.tokenizer import build_tokenizer, encode_chars

MIXED = "قال: «الذكاء الاصطناعي» 2026 ٣٤٥\nGPU، ﴿بِسْمِ اللَّهِ﴾ 🙂 ✓"


def test_vocab_has_unique_tokens_and_starts_with_specials():
    assert len(VOCAB) == len(set(VOCAB))
    assert VOCAB[:5] == ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def test_fast_path_matches_tokenizer_and_offsets_are_characters():
    tokenizer = build_tokenizer()
    encoding = tokenizer(MIXED, return_offsets_mapping=True)
    ids = encoding["input_ids"]
    assert ids[0] == TOKEN_TO_ID[CLS] and ids[-1] == TOKEN_TO_ID[SEP]
    assert ids[1:-1] == encode_chars(MIXED)
    offsets = encoding["offset_mapping"][1:-1]
    assert offsets == [(i, i + 1) for i in range(len(MIXED))]


def test_unknown_characters_become_unk():
    ids = encode_chars("🙂✓")
    assert ids == [TOKEN_TO_ID[UNK]] * 2


def test_save_and_reload_with_auto_tokenizer(tmp_path):
    build_tokenizer().save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert reloaded(MIXED)["input_ids"][1:-1] == encode_chars(MIXED)
    assert reloaded.pad_token_id == TOKEN_TO_ID["[PAD]"]
    assert reloaded.mask_token_id == TOKEN_TO_ID["[MASK]"]
