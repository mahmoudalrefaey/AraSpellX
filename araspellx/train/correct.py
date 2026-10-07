"""Train the correction model (one edit label per character) from the pretrained encoder.

Starts from the masked-LM encoder of train/pretrain.py, adds the label
classifier and trains on windows drawn from train/correction_data.py.
Every --eval_every steps it corrects fixed development sets and reports
precision, recall, F0.5 and damage (changes to correct words). Saves a full
checkpoint every --ckpt_minutes; rerunning the same command resumes.

Short check run, then the full run:

    python -m araspellx.train.correct --out artifacts/correct_check --max_steps 600 --eval_every 300
    python -m araspellx.train.correct --out artifacts/correct --max_steps 30000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
import zlib
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from araspellx.correct.decode import correct_texts
from araspellx.data.labels import LabelVocab
from araspellx.eval.metrics import score
from araspellx.model.bert import BertForMaskedLM, BertForTokenClassification
from araspellx.noise.typed import typed_noise
from araspellx.train.correction_data import (
    Mixture, Paragraphs, batch, build_vocab, load_pairs, normalized_pairs, read_rows)
from araspellx.text.charset import PAD, TOKEN_TO_ID
from araspellx.text.tokenizer import build_tokenizer

DEV_SIZE = 300


def is_dev(pair_id_or_text: str) -> bool:
    """1% of pairs, chosen by hash, are kept for development and never trained on."""
    return zlib.crc32(pair_id_or_text.encode()) % 100 == 0


def load_split_pairs(paths):
    """Training and development pairs (1% of pair ids, by hash, go to development)."""
    rows = read_rows(paths)
    return (normalized_pairs([r for r in rows if not is_dev(r["id"])]),
            normalized_pairs([r for r in rows if is_dev(r["id"])]))


def dev_sets(valid: Paragraphs, ocr_dev, yarmouk_dev):
    rng = random.Random(1234)
    clean = [valid.sample(rng) for _ in range(DEV_SIZE)]
    return {
        "typed": [(typed_noise(c, rng), c) for c in clean],
        "clean": [(c, c) for c in [valid.sample(rng) for _ in range(DEV_SIZE)]],
        "ocr_render": ocr_dev[:DEV_SIZE],
        "yarmouk_real": yarmouk_dev[:DEV_SIZE],
    }


def evaluate(model, sets, vocab, device):
    model.eval()
    report = {}
    for name, pairs in sets.items():
        if not pairs:
            continue
        sources = [n for n, _ in pairs]
        outputs = correct_texts(model, sources, vocab, device)
        s = score(sources, outputs, [c for _, c in pairs], category=None)
        report[name] = {"precision": round(s.precision, 3), "recall": round(s.recall, 3),
                        "f0.5": round(s.f05, 3), "damage": round(s.damage, 4),
                        "cer_in": round(s.rates()["cer_in"], 4), "cer_out": round(s.rates()["cer_out"], 4)}
    model.train()
    return report


def learning_rate(step, peak, warmup, total):
    if step < warmup:
        return peak * step / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return peak * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * progress)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pretrained", type=Path, default=Path("artifacts/pretrain/model"))
    parser.add_argument("--text", type=Path, default=Path("data/v1/pretrain"))
    parser.add_argument("--ocr_pairs", type=Path, nargs="+",
                        default=[Path("data/v1/pairs/ocr_render.jsonl"), Path("data/v1/pairs/yarmouk_train.jsonl")])
    parser.add_argument("--yarmouk_dev", type=Path, default=Path("data/v1/testsets/yarmouk_dev.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/correct"))
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--ckpt_minutes", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    paragraphs = Paragraphs(args.text / "train.bin")
    ocr_train, ocr_dev = load_split_pairs([p for p in args.ocr_pairs if p.exists()])
    yarmouk_dev = load_pairs([args.yarmouk_dev]) if args.yarmouk_dev.exists() else []
    mixture = Mixture(paragraphs, ocr_train)
    print(f"sources {dict(zip(mixture.sources, [round(w, 2) for w in mixture.weights]))}, "
          f"{len(ocr_train)} OCR pairs for training, {len(ocr_dev)} for development")

    vocab_path = args.out / "labels.json"
    if vocab_path.exists():
        vocab = LabelVocab.load(vocab_path)
    else:
        vocab = build_vocab(mixture, samples=20000, seed=args.seed)
        vocab.save(vocab_path)
    print(f"{len(vocab)} labels")

    pretrained = BertForMaskedLM.from_pretrained(args.pretrained)
    config = pretrained.config
    config.num_labels = len(vocab)
    config.id2label = dict(enumerate(vocab.labels))
    config.label2id = {label: i for i, label in enumerate(vocab.labels)}
    model = BertForTokenClassification(config)
    model.bert.load_state_dict(pretrained.bert.state_dict())
    model.to(device).train()
    del pretrained

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
    rng = random.Random(args.seed)
    step = 0
    ckpt = args.out / "last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        rng.setstate(state["rng"])
        print(f"Resumed from {ckpt} at step {step}")

    def save():
        partial = ckpt.with_suffix(".partial")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "rng": rng.getstate(), "args": {k: str(v) for k, v in vars(args).items()}}, partial)
        partial.replace(ckpt)

    sets = dev_sets(Paragraphs(args.text / "valid.bin"), ocr_dev, yarmouk_dev)
    writer = SummaryWriter(args.out / "logs")
    last_ckpt = started = time.time()
    start_step, loss_avg = step, None
    while step < args.max_steps:
        inputs, labels = batch(mixture, vocab, args.batch_size, rng)
        inputs, labels = inputs.to(device), labels.to(device)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step, args.lr, args.warmup, args.max_steps)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs, (inputs != TOKEN_TO_ID[PAD]).long())
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        step += 1
        loss_avg = loss.item() if loss_avg is None else 0.98 * loss_avg + 0.02 * loss.item()

        if step % 100 == 0:
            rate = (step - start_step) / (time.time() - started)
            print(f"step {step:6d}  loss {loss_avg:.4f}  {rate:.2f} steps/s  "
                  f"ETA {(args.max_steps - step) / rate / 3600:.1f} h  "
                  f"GPU {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            writer.add_scalar("train/loss", loss_avg, step)
        if step % args.eval_every == 0 or step == args.max_steps:
            report = evaluate(model, sets, vocab, device)
            print(f"== step {step}")
            for name, values in report.items():
                print(f"   {name:13s} {values}")
                for key, value in values.items():
                    writer.add_scalar(f"{name}/{key}", value, step)
            with open(args.out / "eval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, **report}) + "\n")
        if time.time() - last_ckpt > args.ckpt_minutes * 60:
            save()
            last_ckpt = time.time()

    save()
    model.save_pretrained(args.out / "model")
    build_tokenizer().save_pretrained(args.out / "model")
    vocab.save(args.out / "model" / "labels.json")
    print(f"Finished: Hugging Face model saved to {args.out / 'model'}")


if __name__ == "__main__":
    main()
