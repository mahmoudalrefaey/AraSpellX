import argparse
import csv
import math
import re
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import torch

from core import constants
from data.tokenizer import get_tokenizer
from models.models import get_model


DEFAULT_CHECKPOINT = r"outdir/checkpoint_epoch_2_step_648966.pt"
DEFAULT_TOKENIZER = r"outdir/tokenizer.json"

# Must match the architecture used during training.
D_MODEL = 512
H = 8
N_LAYERS = 4
HIDDEN_SIZE = 256
DROPOUT = 0.1

# Must match the data settings used during training (core/args.py defaults).
MAX_LEN = 128            # --max_len: longest clean sentence, in characters
DISTORTION_RATIO = 0.1   # --distortion_ratio

# ArabicData._get_distorted (data/data.py) pads every encoder input
# (SOS + chars + EOS) to int(ratio * max_len) + max_len + 1 tokens.
# Checkpoints trained without padding masks attend to these padding
# positions, so inference must reproduce exactly the same input layout.
ENCODER_LEN = int(DISTORTION_RATIO * MAX_LEN) + MAX_LEN + 1

# Arabic text is corrected in chunks no longer than the longest
# training sentence (decoder positions beyond it were never trained).
MAX_CHUNK_CHARS = MAX_LEN

BATCH_SIZE = 64

# Arabic presentation forms (ligatures such as U+FEFB) -> base letters.
PRESENTATION_FORMS = re.compile('[\uFB50-\uFDFF\uFE70-\uFEFE]')

# Look-alike code points outside the vocabulary -> the vocabulary letter.
LOOKALIKES = str.maketrans({
    '\u06CC': '\u064A',  # FARSI YEH -> YEH
    '\u06A9': '\u0643',  # KEHEH -> KAF
    '\u06C1': '\u0647',  # HEH GOAL -> HEH
    '\u06BE': '\u0647',  # HEH DOACHASHMEE -> HEH
    '\u06D5': '\u0647',  # AE -> HEH
    '\u06C3': '\u0629',  # TEH MARBUTA GOAL -> TEH MARBUTA
    '\u0671': '\u0627',  # ALEF WASLA -> ALEF
})

# Removed like in the data pipeline: harakat, superscript alef, tatweel,
# zero-width and bidi control characters.
REMOVED_CHARS = re.compile(
    '[' + constants.ARABIC_HARAKAT +
    '\u0670\u0640\u200B-\u200F\u202A-\u202E\u2066-\u2069\uFEFF]'
)

ARABIC_WORD = re.compile(f'[{constants.ARABIC_CHARS}]+')


def load_model(checkpoint_path, tokenizer_path, device):
    print("=" * 70)
    print("Loading AraSpellX")
    print("=" * 70)

    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found:\n{tokenizer_path}"
        )

    # ---------------------------------------------------------
    # Build the exact same tokenizer used during training.
    # ---------------------------------------------------------
    tokenizer = get_tokenizer(SimpleNamespace(tokenizer_path=tokenizer_path))

    print(f"Tokenizer: {tokenizer_path}")
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"SOS id: {tokenizer.special_tokens.sos_id}")
    print(f"EOS id: {tokenizer.special_tokens.eos_id}")
    print(f"PAD id: {tokenizer.special_tokens.pad_id}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    # ---------------------------------------------------------
    # Build the exact same Transformer architecture.
    # Checkpoints saved before padding masks existed have no
    # 'mask_padding' entry and were trained without them.
    # ---------------------------------------------------------
    model_args = SimpleNamespace(
        model="transformer",
        d_model=D_MODEL,
        h=H,
        n_layers=N_LAYERS,
        p_dropout=DROPOUT,
        hidden_size=HIDDEN_SIZE,
        mask_padding=checkpoint.get("mask_padding", False),
    )

    model = get_model(
        model_args,
        rank=0,
        voc_size=tokenizer.vocab_size,
        pad_idx=tokenizer.special_tokens.pad_id,
    )

    model.to(device)

    # ---------------------------------------------------------
    # Load trained weights.
    # ---------------------------------------------------------
    state_dict = checkpoint["model"]

    # Handle possible DataParallel "module." prefix.
    state_dict = {
        key.replace("module.", "", 1): value
        for key, value in state_dict.items()
    }

    model.load_state_dict(state_dict, strict=True)

    model.eval()

    print(f"Checkpoint: {checkpoint_path}")

    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch']}")

    if "global_step" in checkpoint:
        print(f"Global step: {checkpoint['global_step']}")

    print(f"Padding masks: {model_args.mask_padding}")
    print(f"Device: {device}")
    print("Model loaded successfully.")
    print("=" * 70)

    return model, tokenizer


def normalize_text(text):
    """Fold Unicode variants into vocabulary letters and drop the marks
    the training data never contained."""
    text = PRESENTATION_FORMS.sub(
        lambda m: unicodedata.normalize("NFKC", m.group()), text
    )
    text = text.translate(LOOKALIKES)
    return REMOVED_CHARS.sub("", text)


def split_spans(text):
    """Split text into (is_arabic, span) pieces.

    Arabic spans are runs of Arabic words separated only by whitespace
    (normalized to single spaces, as in the training data). Everything
    else - punctuation, digits, Latin words, surrounding whitespace -
    is returned verbatim so it never reaches the model, whose vocabulary
    does not contain it.
    """
    spans = []
    words = []
    run_end = 0
    pos = 0

    for match in ARABIC_WORD.finditer(text):
        if words and text[run_end:match.start()].isspace():
            words.append(match.group())
            run_end = match.end()
            continue

        if words:
            spans.append((True, " ".join(words)))
            pos = run_end

        if match.start() > pos:
            spans.append((False, text[pos:match.start()]))

        words = [match.group()]
        run_end = match.end()

    if words:
        spans.append((True, " ".join(words)))
        pos = run_end

    if pos < len(text):
        spans.append((False, text[pos:]))

    return spans


def split_into_chunks(text, max_chars=MAX_CHUNK_CHARS):
    """Split an Arabic span at word boundaries into balanced chunks
    of at most max_chars characters."""
    if len(text) <= max_chars:
        return [text]

    words = []
    for word in text.split(" "):
        # A single word longer than max_chars is cut into pieces.
        words.extend(
            word[i:i + max_chars] for i in range(0, len(word), max_chars)
        )

    target = math.ceil(len(text) / math.ceil(len(text) / max_chars))

    chunks = []
    current = []
    current_len = 0

    for word in words:
        added = len(word) + (1 if current else 0)

        if current and (
            current_len + added > max_chars or current_len >= target
        ):
            chunks.append(" ".join(current))
            current = []
            current_len = 0
            added = len(word)

        current.append(word)
        current_len += added

    if current:
        chunks.append(" ".join(current))

    return chunks


@torch.inference_mode()
def correct_batch(
    model,
    tokenizer,
    texts,
    device,
    max_gen_len=MAX_LEN + 1,
):
    """
    Correct a batch of Arabic-only texts (each at most MAX_CHUNK_CHARS
    characters) using greedy autoregressive decoding.

    The model was trained as:

        SOS + distorted text + EOS + PAD...   (ENCODER_LEN tokens)
                  ↓
             Transformer
                  ↓
        corrected text + EOS
    """
    if not texts:
        return []

    sos_id = tokenizer.special_tokens.sos_id
    eos_id = tokenizer.special_tokens.eos_id
    pad_id = tokenizer.special_tokens.pad_id

    # ---------------------------------------------------------
    # Encoder input, built exactly like ArabicData._get_distorted.
    # mask: False = real token, True = padding.
    # ---------------------------------------------------------
    source = []
    mask = []

    for text in texts:
        ids = tokenizer.tokenize(text, add_sos=True, add_eos=True)

        if len(ids) > ENCODER_LEN:
            raise ValueError(
                f"Chunk of {len(text)} characters does not fit the "
                f"{ENCODER_LEN}-token encoder input."
            )

        n_pad = ENCODER_LEN - len(ids)
        source.append(ids + [pad_id] * n_pad)
        mask.append([False] * len(ids) + [True] * n_pad)

    encoded = torch.tensor(source, dtype=torch.long, device=device)
    encoder_mask = torch.tensor(mask, dtype=torch.bool, device=device)

    # ---------------------------------------------------------
    # Decoder starts with SOS.
    #
    # Transformer.predict() encodes the source on the first call
    # (decoder length == 1) and returns the encoded source, which
    # is reused afterwards.
    # ---------------------------------------------------------
    decoder = torch.full(
        (len(texts), 1),
        sos_id,
        dtype=torch.long,
        device=device,
    )
    finished = torch.zeros(len(texts), dtype=torch.bool, device=device)

    for _ in range(max_gen_len):
        encoded, log_probs, _ = model.predict(
            dec_inp=decoder,
            enc_inp=encoded,
            enc_mask=encoder_mask,
        )

        next_ids = torch.argmax(log_probs[:, -1, :], dim=-1)
        decoder = torch.cat([decoder, next_ids.unsqueeze(1)], dim=1)

        finished |= next_ids == eos_id
        if finished.all():
            break

    # ---------------------------------------------------------
    # Convert generated IDs back to characters.
    # ---------------------------------------------------------
    results = []

    for ids in decoder[:, 1:].tolist():
        # Remove EOS and anything after it.
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]

        ids = [i for i in ids if i not in (sos_id, pad_id)]
        results.append("".join(tokenizer.ids2tokens(ids)))

    return results


def correct_texts(
    model,
    tokenizer,
    texts,
    device,
    max_gen_len=MAX_LEN + 1,
    batch_size=BATCH_SIZE,
):
    """Correct a list of raw texts of any length. Only Arabic spans are
    sent to the model; all other characters are kept as written."""
    all_spans = [split_spans(normalize_text(text.strip())) for text in texts]

    chunks = [
        chunk
        for spans in all_spans
        for is_arabic, span in spans if is_arabic
        for chunk in split_into_chunks(span)
    ]

    corrected = []
    for i in range(0, len(chunks), batch_size):
        corrected += correct_batch(
            model, tokenizer, chunks[i:i + batch_size], device, max_gen_len
        )
    corrected = iter(corrected)

    results = []
    for spans in all_spans:
        pieces = []
        for is_arabic, span in spans:
            if is_arabic:
                span = " ".join(
                    next(corrected) for _ in split_into_chunks(span)
                )
            pieces.append(span)
        results.append("".join(pieces))

    return results


def correct_text(
    model,
    tokenizer,
    text,
    device,
    max_gen_len=MAX_LEN + 1,
):
    """Correct one Arabic text of any length."""
    return correct_texts(model, tokenizer, [text], device, max_gen_len)[0]


def evaluate_csv(model, tokenizer, device, args):
    """Report CER/WER of the inference path on a CSV of input/reference pairs."""
    import editdistance

    with open(args.eval_csv, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if args.limit is not None:
        rows = rows[:args.limit]

    inputs = [row[args.input_col] for row in rows]
    references = [row[args.ref_col] for row in rows]

    start = time.perf_counter()
    predictions = correct_texts(
        model, tokenizer, inputs, device, args.max_gen_len
    )
    elapsed = time.perf_counter() - start

    def rates(hypotheses):
        char_edits = sum(
            editdistance.eval(h, r) for h, r in zip(hypotheses, references)
        )
        word_edits = sum(
            editdistance.eval(h.split(), r.split())
            for h, r in zip(hypotheses, references)
        )
        chars = sum(len(r) for r in references)
        words = sum(len(r.split()) for r in references)
        exact = sum(h == r for h, r in zip(hypotheses, references))
        return char_edits / chars, word_edits / words, exact / len(references)

    print(f"Evaluated {len(rows)} rows of {args.eval_csv} in {elapsed:.1f}s")
    print("Input (no correction): CER=%.4f WER=%.4f exact=%.3f" % rates(inputs))
    print("Model output:          CER=%.4f WER=%.4f exact=%.3f" % rates(predictions))

    for inp, ref, pred in list(zip(inputs, references, predictions))[:args.show]:
        print()
        print(f"Input:     {inp}")
        print(f"Reference: {ref}")
        print(f"Output:    {pred}")


def interactive_mode(model, tokenizer, device):
    print()
    print("=" * 70)
    print("AraSpellX Interactive Inference")
    print("=" * 70)
    print("Enter Arabic text to correct it.")
    print("Type 'exit' or 'quit' to stop.")
    print("=" * 70)

    while True:
        try:
            text = input("\nInput: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if text.lower() in {"exit", "quit"}:
            break

        if not text:
            continue

        try:
            start = time.perf_counter()

            corrected = correct_text(
                model=model,
                tokenizer=tokenizer,
                text=text,
                device=device,
            )

            elapsed = time.perf_counter() - start

            print(f"Output: {corrected}")
            print(f"Time: {elapsed:.3f}s")

        except Exception as e:
            print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="AraSpellX Arabic spelling correction inference"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to AraSpellX checkpoint",
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default=DEFAULT_TOKENIZER,
        help="Path to tokenizer.json",
    )

    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Correct one text directly instead of interactive mode",
    )

    parser.add_argument(
        "--max_gen_len",
        type=int,
        default=MAX_LEN + 1,
        help="Maximum number of generated tokens per chunk (including EOS)",
    )

    parser.add_argument(
        "--eval_csv",
        type=str,
        default=None,
        help="Evaluate the inference path on a CSV instead of interactive mode",
    )

    parser.add_argument(
        "--input_col",
        type=str,
        default="distorted_0.1",
        help="Input column of --eval_csv",
    )

    parser.add_argument(
        "--ref_col",
        type=str,
        default="clean",
        help="Reference column of --eval_csv",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of --eval_csv rows to evaluate (default: all)",
    )

    parser.add_argument(
        "--show",
        type=int,
        default=0,
        help="Number of --eval_csv examples to print",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU inference",
    )

    args = parser.parse_args()

    if args.cpu:
        device = torch.device("cpu")
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    model, tokenizer = load_model(
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
        device=device,
    )

    if args.eval_csv is not None:
        evaluate_csv(model, tokenizer, device, args)

    elif args.text is not None:
        start = time.perf_counter()

        corrected = correct_text(
            model=model,
            tokenizer=tokenizer,
            text=args.text,
            device=device,
            max_gen_len=args.max_gen_len,
        )

        elapsed = time.perf_counter() - start

        print()
        print(f"Input:  {args.text}")
        print(f"Output: {corrected}")
        print(f"Time:   {elapsed:.3f}s")

    else:
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            device=device,
        )


if __name__ == "__main__":
    main()
