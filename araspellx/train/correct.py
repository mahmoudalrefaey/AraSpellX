"""Train the correction model (one edit label per character) from the pretrained encoder.

Starts from the masked-LM encoder of train/pretrain.py, adds the label
classifier and trains on windows drawn from train/correction_data.py.
Every --eval_every steps it corrects fixed development sets and reports
precision, recall, F0.5 and damage (changes to correct words).

Safety, as in pretraining: runaway attention scores are capped
(train/stability.py); the model with the best mean F0.5 over the error
development sets is kept in best_model/ (and best.pt); training stops by
itself if that F0.5 falls --stop_drop below the best. A full checkpoint is
saved every --ckpt_minutes; rerunning the same command resumes.

Short check run, then the full run:

    python -m araspellx.train.correct --out artifacts/correct_check --max_steps 1000 --eval_every 500
    python -m araspellx.train.correct --out artifacts/correct --max_steps 30000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
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
from araspellx.train.progress import Prefetcher, Progress, gpu_memory_gb, precision_settings
from araspellx.train.stability import cap_attention, max_attention_scores

DEV_SIZE = 300
ERROR_SETS = ("typed", "ocr_render", "yarmouk_real", "wiki_edits")  # development sets with errors to fix


def is_dev(pair_id_or_text: str) -> bool:
    """1% of pairs, chosen by hash, are kept for development and never trained on."""
    return zlib.crc32(pair_id_or_text.encode()) % 100 == 0


def load_split_pairs(paths):
    """Training and development pairs (1% of pair ids, by hash, go to development)."""
    rows = read_rows(paths)
    return (normalized_pairs([r for r in rows if not is_dev(r["id"])]),
            normalized_pairs([r for r in rows if is_dev(r["id"])]))


def dev_sets(valid: Paragraphs, ocr_dev, yarmouk_dev, edits_dev=()):
    rng = random.Random(1234)
    clean = [valid.sample(rng) for _ in range(DEV_SIZE)]
    return {
        "typed": [(typed_noise(c, rng), c) for c in clean],
        "clean": [(c, c) for c in [valid.sample(rng) for _ in range(DEV_SIZE)]],
        "ocr_render": ocr_dev[:DEV_SIZE],
        "yarmouk_real": yarmouk_dev[:DEV_SIZE],
        "wiki_edits": list(edits_dev)[:DEV_SIZE],
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
    parser.add_argument("--edits", type=Path, default=Path("data/v1/pairs/wiki_edits_train.jsonl"),
                        help="real spelling fixes mined from Wikipedia's history (testsets/edits.py), if present")
    parser.add_argument("--out", type=Path, default=Path("artifacts/correct"))
    parser.add_argument("--max_steps", type=int, default=30000)
    parser.add_argument("--batch_size", type=int, default=32, help="512-character windows per step")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--ckpt_minutes", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="auto: bf16 on GPUs that support it (RTX 30xx, A100), else fp16 (T4, V100)")
    parser.add_argument("--attention_cap", type=float, default=50.0,
                        help="largest attention score allowed per head (0: no cap)")
    parser.add_argument("--cap_every", type=int, default=50, help="steps between attention-cap checks")
    parser.add_argument("--stop_drop", type=float, default=0.15,
                        help="stop when mean F0.5 falls this far below the best (absolute)")
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast, scaler, precision = precision_settings(args.precision)
    args.out.mkdir(parents=True, exist_ok=True)
    messages = []  # logged once the progress display exists
    paragraphs = Paragraphs(args.text / "train.bin")
    ocr_train, ocr_dev = load_split_pairs([p for p in args.ocr_pairs if p.exists()])
    yarmouk_dev = load_pairs([args.yarmouk_dev]) if args.yarmouk_dev.exists() else []
    edits_train, edits_dev = load_split_pairs([args.edits]) if args.edits.exists() else ([], [])
    mixture = Mixture(paragraphs, ocr_train, edits_train)
    messages.append(f"sources {dict(zip(mixture.sources, [round(w, 2) for w in mixture.weights]))} | "
                    f"{len(ocr_train)} OCR pairs for training, {len(ocr_dev)} for development | "
                    f"{len(edits_train)} real spelling-fix pairs for training, {len(edits_dev)} for development")

    vocab_path = args.out / "labels.json"
    if vocab_path.exists():
        vocab = LabelVocab.load(vocab_path)
    else:
        vocab = build_vocab(mixture, samples=20000, seed=args.seed)
        vocab.save(vocab_path)
    messages.append(f"{len(vocab)} labels")

    pretrained = BertForMaskedLM.from_pretrained(args.pretrained)
    config = pretrained.config
    config.num_labels = len(vocab)
    config.id2label = dict(enumerate(vocab.labels))
    config.label2id = {label: i for i, label in enumerate(vocab.labels)}
    model = BertForTokenClassification(config)
    model.bert.load_state_dict(pretrained.bert.state_dict())
    model.to(device).train()
    del pretrained

    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    no_decay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.98))
    rng = random.Random(args.seed)
    step, best = 0, None  # best: (mean F0.5 over the error sets, step)
    ckpt = args.out / "last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        rng.setstate(state["rng"])
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best = state.get("best")
        del state

    progress = Progress(args.max_steps, step, args.out / "train.log", "correct")
    progress.log(f"Resumed from {ckpt} at step {step}" if step else f"Starting at step 0 -> {args.out}")
    for message in messages:
        progress.log(message)
    progress.log(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters | "
                 f"{args.max_steps} steps x {args.batch_size} windows | precision {precision} | "
                 f"GPU {torch.cuda.get_device_name()}")

    def save(path: Path, message: str):
        partial = path.with_suffix(".partial")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "rng": rng.getstate(), "scaler": scaler.state_dict() if scaler else None, "best": best,
                    "args": {k: str(v) for k, v in vars(args).items()}}, partial)
        partial.replace(path)
        progress.log(message)

    def save_model(directory: Path):
        model.save_pretrained(directory)
        build_tokenizer().save_pretrained(directory)
        vocab.save(directory / "labels.json")

    # A fixed set of training windows on which attention scores are measured.
    probe, _ = batch(mixture, vocab, 8, random.Random(12345))
    probe = probe.to(device)
    probe_mask = (probe != TOKEN_TO_ID[PAD]).long()
    capped_heads, capped_recently = set(), {}

    def check_attention() -> None:
        for layer, head, score in cap_attention(model, probe, args.attention_cap, probe_mask):
            key = (layer, head)
            capped_recently[key] = max(capped_recently.get(key, 0.0), score)
            if key not in capped_heads:
                capped_heads.add(key)
                progress.log(f"attention cap: layer {layer} head {head} reached a score of {score:,.0f}; "
                             f"scaled back to {args.attention_cap:.0f}")

    sets = dev_sets(Paragraphs(args.text / "valid.bin"), ocr_dev, yarmouk_dev, edits_dev)
    writer = SummaryWriter(args.out / "logs")
    if args.attention_cap:
        check_attention()
    batches = Prefetcher(lambda: batch(mixture, vocab, args.batch_size, rng))
    last_ckpt = started = time.time()
    start_step, loss_avg, acc_avg, metrics = step, None, None, {}
    while step < args.max_steps:
        inputs, labels = batches.next()
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        lr = learning_rate(step, args.lr, args.warmup, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast():
            logits = model(inputs, (inputs != TOKEN_TO_ID[PAD]).long())
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        step += 1
        if args.attention_cap and step % args.cap_every == 0:
            check_attention()

        if step % 10 == 0:  # reading values back from the GPU every step would slow training
            # accuracy on characters whose label is not "keep": how often the model finds the fix
            valid = labels != -100
            edits = valid & (labels != 0)  # label 0 is KEEP
            predicted = logits.argmax(-1)
            edit_acc = (predicted[edits] == labels[edits]).float().mean().item() if edits.any() else 0.0
            loss_avg = loss.item() if loss_avg is None else 0.9 * loss_avg + 0.1 * loss.item()
            acc_avg = edit_acc if acc_avg is None else 0.9 * acc_avg + 0.1 * edit_acc
            windows = (step - start_step) * args.batch_size / (time.time() - started)
            metrics = {"loss": loss_avg, "edit acc": acc_avg, "lr": lr, "grad": grad_norm.item(),
                       "win/s": windows, "GPU GB": gpu_memory_gb()}
        progress.step(metrics, write_every=100, step=step)
        if step % 100 == 0 and metrics:
            writer.add_scalar("train/loss", loss_avg, step)
            writer.add_scalar("train/edit_accuracy", acc_avg, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", metrics["grad"], step)
        if step % args.eval_every == 0 or step == args.max_steps:
            progress.log(f"evaluating development sets at step {step} ...")
            report = evaluate(model, sets, vocab, device)
            scores = max_attention_scores(model, probe, probe_mask).amax(dim=1).tolist()  # per layer
            capped = ", ".join(f"layer {l} head {h} (up to {s:,.0f})" for (l, h), s in sorted(capped_recently.items()))
            capped_recently.clear()
            errors = [report[name]["f0.5"] for name in ERROR_SETS if name in report]
            mean_f05 = sum(errors) / len(errors)
            progress.log(f"eval step {step}: mean F0.5 over the error sets {mean_f05:.3f}\n" + "\n".join(
                f"    {name:13s} " + "  ".join(f"{k} {v}" for k, v in values.items())
                for name, values in report.items())
                + f"\n    largest attention score per layer: {' '.join(f'{s:.0f}' for s in scores)}"
                + (f" | capped since last evaluation: {capped}" if capped else ""))
            for name, values in report.items():
                for key, value in values.items():
                    writer.add_scalar(f"{name}/{key}", value, step)
            writer.add_scalar("dev/mean_f05", mean_f05, step)
            with open(args.out / "eval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "mean_f0.5": mean_f05, **report,
                                    "max_attention_scores": scores}) + "\n")
            if best is None or mean_f05 > best[0]:
                best = (mean_f05, step)
                save(args.out / "best.pt", f"new best mean F0.5: {mean_f05:.3f} (saved best.pt and best_model)")
                save_model(args.out / "best_model")
            elif mean_f05 < best[0] - args.stop_drop:
                progress.log(f"STOPPED: mean F0.5 {mean_f05:.3f} is more than {args.stop_drop} below the best "
                             f"({best[0]:.3f} at step {best[1]}). The best model is in {args.out / 'best_model'} "
                             f"and its full checkpoint in {args.out / 'best.pt'}.")
                progress.close()
                sys.exit(1)
        if time.time() - last_ckpt > args.ckpt_minutes * 60:
            save(ckpt, f"checkpoint saved at step {step}")
            last_ckpt = time.time()

    save(ckpt, f"checkpoint saved at step {step}")
    save_model(args.out / "model")
    progress.log(f"Finished: Hugging Face model saved to {args.out / 'model'}")
    progress.close()


if __name__ == "__main__":
    main()
