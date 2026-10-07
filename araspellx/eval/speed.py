"""CPU speed benchmark for release gate G6 (words per second on CPU).

The model reads text in overlapping windows of WINDOW_TOKENS tokens
([CLS] + characters + [SEP]); each window keeps MARGIN characters of
context on both sides that belong to the neighbouring windows. Speed
does not depend on which characters are read, only on how many, so the
benchmark can run on untrained models of any size.

    python -m araspellx.eval.speed --configs 6x384 8x384 8x512 --threads 4
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import List

import numpy as np

WINDOW_TOKENS = 512
MARGIN = 64
CONTENT = WINDOW_TOKENS - 2


def window_starts(n_chars: int, content: int = CONTENT, margin: int = MARGIN) -> List[int]:
    """Start offsets of the windows that cover n_chars characters."""
    if n_chars <= content:
        return [0]
    stride = content - 2 * margin
    starts = list(range(0, n_chars - content, stride))
    starts.append(n_chars - content)
    return starts


def benchmark(session, text: str, vocab_size: int, batch_size: int = 16) -> float:
    """Return words per second for one pass of the model over text."""
    ids = np.array([3 + ord(c) % (vocab_size - 3) for c in text], dtype=np.int64)
    batch, mask = [], []
    windows = []
    for start in window_starts(len(ids)):
        chunk = ids[start:start + CONTENT]
        row = np.zeros(WINDOW_TOKENS, dtype=np.int64)
        row[0], row[1:len(chunk) + 1], row[len(chunk) + 1] = 1, chunk, 2
        row_mask = np.zeros(WINDOW_TOKENS, dtype=np.int64)
        row_mask[:len(chunk) + 2] = 1
        windows.append((row, row_mask))

    started = time.perf_counter()
    for i in range(0, len(windows), batch_size):
        part = windows[i:i + batch_size]
        session.run(None, {
            "input_ids": np.stack([w[0] for w in part]),
            "attention_mask": np.stack([w[1] for w in part]),
        })
    elapsed = time.perf_counter() - started
    return len(text.split()) / elapsed


def export_random_model(layers: int, hidden: int, path: Path, vocab_size: int = 300,
                        num_labels: int = 150) -> int:
    """Export an untrained BERT token classifier to ONNX; return its parameter count."""
    import torch
    from transformers import BertConfig, BertForTokenClassification

    config = BertConfig(
        vocab_size=vocab_size,
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=hidden // 64,
        intermediate_size=4 * hidden,
        max_position_embeddings=WINDOW_TOKENS,
        type_vocab_size=1,
        num_labels=num_labels,
        attn_implementation="eager",
    )
    model = BertForTokenClassification(config).eval()
    dummy = torch.ones(2, WINDOW_TOKENS, dtype=torch.long)
    torch.onnx.export(
        model, (dummy, dummy), str(path),
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={name: {0: "batch", 1: "sequence"}
                      for name in ["input_ids", "attention_mask", "logits"]},
        opset_version=17,
    )
    return sum(p.numel() for p in model.parameters())


def load_text(words: int, source: Path) -> str:
    """Join clean sentences from a dataset CSV into one text of ~words words."""
    text, count = [], 0
    with open(source, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            text.append(row["clean"])
            count += len(row["clean"].split())
            if count >= words:
                break
    return " ".join(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--configs", nargs="+", default=["6x384", "8x384", "6x512", "8x512"],
                        help="LAYERSxHIDDEN, e.g. 8x512")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--words", type=int, default=20000)
    parser.add_argument("--text_csv", type=Path, default=Path("data/dataset/test.csv"))
    parser.add_argument("--workdir", type=Path, default=Path("artifacts/speed"))
    args = parser.parse_args()

    import onnxruntime as ort
    from onnxruntime.quantization import QuantType, quantize_dynamic

    args.workdir.mkdir(parents=True, exist_ok=True)
    text = load_text(args.words, args.text_csv)
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1

    print(f"{len(text.split())} words, {len(text)} characters, {args.threads} threads")
    print(f"{'config':8s} {'params':>8s} {'fp32 w/s':>9s} {'int8 w/s':>9s}")
    for spec in args.configs:
        layers, hidden = map(int, spec.split("x"))
        fp32 = args.workdir / f"bert_{spec}.onnx"
        int8 = args.workdir / f"bert_{spec}.int8.onnx"
        params = export_random_model(layers, hidden, fp32)
        quantize_dynamic(str(fp32), str(int8), weight_type=QuantType.QInt8)
        speeds = []
        for path in (fp32, int8):
            session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
            benchmark(session, text[:5000], 300)  # warm-up
            speeds.append(benchmark(session, text, 300))
        print(f"{spec:8s} {params / 1e6:7.1f}M {speeds[0]:9.0f} {speeds[1]:9.0f}")


if __name__ == "__main__":
    main()
