"""Pretrain AraSpellX's character encoder with masked character prediction.

Reads the token stream from data/pretrain_corpus.py, masks 15% of the
characters of each 512-token window (half as whole words, half as random
1-5 character spans) and trains BertForMaskedLM to restore them. Saves a
full checkpoint every --ckpt_minutes; rerunning the same command resumes.

Short check run (~5 minutes), then the full run:

    python -m araspellx.train.pretrain --out artifacts/pretrain_check --max_steps 600 --eval_every 200
    python -m araspellx.train.pretrain --out artifacts/pretrain --max_steps 40000
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from araspellx.model.bert import BertForMaskedLM, make_config
from araspellx.text.charset import ARABIC, ARABIC_LETTERS, CLS, DIACRITICS, MASK, PAD, SEP, TOKEN_TO_ID, VOCAB
from araspellx.text.tokenizer import build_tokenizer, encode_chars
from araspellx.train.progress import Prefetcher, Progress, gpu_memory_gb, precision_settings

WINDOW = 512
CONTENT = WINDOW - 2
LETTER_IDS = np.array([TOKEN_TO_ID[c] for c in ARABIC_LETTERS])  # random replacements
IS_LETTER = np.zeros(len(VOCAB), dtype=bool)
IS_LETTER[[TOKEN_TO_ID[c] for c in ARABIC | set(DIACRITICS) if c in TOKEN_TO_ID]] = True

EXAMPLES = [  # (sentence, word to hide) shown at every evaluation
    ("ذهب الطلاب إلى الجامعة لحضور المحاضرات في الصباح", "الجامعة"),
    ("تعمل الشركة على تطوير نظام جديد لمعالجة البيانات", "البيانات"),
    ("قال الشاعر في قصيدته إن الوطن أغلى من كل شيء", "قصيدته"),
    ("وفي سنة خمس وثلاثين ومائة توفي الإمام بالمدينة", "توفي"),
]


def mask_window(ids: np.ndarray, rng: np.random.Generator, rate: float = 0.15):
    """Mask ~rate of the characters of one window: half whole words, half short spans."""
    n = len(ids)
    target = np.zeros(n, dtype=bool)
    budget = int(rate * n)
    letter = IS_LETTER[ids]
    starts = np.flatnonzero(letter & ~np.concatenate([[False], letter[:-1]]))
    rng.shuffle(starts)
    for start in starts:  # whole words
        if target.sum() >= budget // 2:
            break
        end = start
        while end < n and letter[end]:
            end += 1
        target[start:end] = True
    while target.sum() < budget:  # random spans
        start, length = rng.integers(0, n), rng.integers(1, 6)
        target[start:start + length] = True
    inputs = ids.copy()
    roll = rng.random(n)
    inputs[target & (roll < 0.8)] = TOKEN_TO_ID[MASK]
    random_pos = target & (roll >= 0.8) & (roll < 0.9)
    inputs[random_pos] = rng.choice(LETTER_IDS, random_pos.sum())
    labels = np.where(target, ids, -100)
    return inputs, labels


class Windows:
    """Random 512-token windows from a uint8 token stream on disk."""

    def __init__(self, path: Path, seed: int) -> None:
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.rng = np.random.default_rng(seed)

    def batch(self, size: int):
        inputs, labels = [], []
        for _ in range(size):
            start = self.rng.integers(0, len(self.data) - CONTENT)
            chunk = np.asarray(self.data[start:start + CONTENT], dtype=np.int64)
            x, y = mask_window(chunk, self.rng)
            inputs.append(np.concatenate([[TOKEN_TO_ID[CLS]], x, [TOKEN_TO_ID[SEP]]]))
            labels.append(np.concatenate([[-100], y, [-100]]))
        return torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(labels))


def learning_rate(step: int, peak: float, warmup: int, total: int) -> float:
    if step < warmup:
        return peak * step / warmup
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


@torch.no_grad()
def evaluate(model, batches, device, autocast):
    model.eval()
    loss_sum = correct = count = 0
    for inputs, labels in batches:
        inputs, labels = inputs.to(device), labels.to(device)
        with autocast():
            logits = model(inputs, (inputs != TOKEN_TO_ID[PAD]).long())
        mask = labels != -100
        loss_sum += F.cross_entropy(logits[mask].float(), labels[mask], reduction="sum").item()
        correct += (logits[mask].argmax(-1) == labels[mask]).sum().item()
        count += mask.sum().item()
    model.train()
    return loss_sum / count, correct / count


@torch.no_grad()
def fill_examples(model, device):
    model.eval()
    lines = []
    for sentence, word in EXAMPLES:
        start = sentence.index(word)
        ids = [TOKEN_TO_ID[CLS]] + encode_chars(sentence) + [TOKEN_TO_ID[SEP]]
        for i in range(start, start + len(word)):
            ids[i + 1] = TOKEN_TO_ID[MASK]
        logits = model(torch.tensor([ids], device=device))
        guess = "".join(VOCAB[i] for i in logits[0, start + 1:start + 1 + len(word)].argmax(-1).tolist())
        lines.append(f"    {word} -> {guess}")
    model.train()
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path("data/v1/pretrain"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/pretrain"))
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--ckpt_minutes", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="auto: bf16 on GPUs that support it (RTX 30xx, A100), else fp16 (T4, V100)")
    args = parser.parse_args()

    device = torch.device("cuda")
    autocast, scaler, precision = precision_settings(args.precision)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    config = make_config(vocab_size=len(VOCAB), layers=args.layers, hidden=args.hidden,
                         max_positions=WINDOW, pad_token_id=TOKEN_TO_ID[PAD])
    model = BertForMaskedLM(config).to(device).train()
    decay = [p for n, p in model.named_parameters() if p.dim() > 1]
    no_decay = [p for n, p in model.named_parameters() if p.dim() <= 1]
    optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": 0.01},
                                   {"params": no_decay, "weight_decay": 0.0}],
                                  lr=args.lr, betas=(0.9, 0.98), eps=1e-6)
    train = Windows(args.data / "train.bin", args.seed)
    valid_batches = [Windows(args.data / "valid.bin", 1234).batch(args.batch_size) for _ in range(8)]

    step = 0
    ckpt = args.out / "last.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        train.rng.bit_generator.state = state["data_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])

    progress = Progress(args.max_steps, step, args.out / "train.log", "pretrain")
    progress.log(f"Resumed from {ckpt} at step {step}" if step else f"Starting at step 0 -> {args.out}")

    def save():
        partial = ckpt.with_suffix(".partial")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "data_rng": train.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "scaler": scaler.state_dict() if scaler else None, "args": vars(args)}, partial)
        partial.replace(ckpt)
        progress.log(f"checkpoint saved at step {step}")

    params = sum(p.numel() for p in model.parameters())
    progress.log(f"{params / 1e6:.1f}M parameters | {len(train.data) / 1e6:,.0f}M training characters | "
                 f"{args.max_steps} steps x {args.batch_size} windows | precision {precision} | "
                 f"GPU {torch.cuda.get_device_name()}")
    writer = SummaryWriter(args.out / "logs")
    batches = Prefetcher(lambda: train.batch(args.batch_size))
    last_ckpt = started = time.time()
    start_step, loss_avg, acc_avg, metrics = step, None, None, {}
    while step < args.max_steps:
        inputs, labels = batches.next()
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        lr = learning_rate(step, args.lr, args.warmup, args.max_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        with autocast():
            logits = model(inputs)
            loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), labels.view(-1),
                                   ignore_index=-100)
        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        step += 1

        if step % 10 == 0:  # reading values back from the GPU every step would slow training
            masked = labels != -100
            accuracy = (logits[masked].argmax(-1) == labels[masked]).float().mean().item()
            loss_avg = loss.item() if loss_avg is None else 0.9 * loss_avg + 0.1 * loss.item()
            acc_avg = accuracy if acc_avg is None else 0.9 * acc_avg + 0.1 * accuracy
            chars = (step - start_step) * args.batch_size * CONTENT / (time.time() - started)
            metrics = {"loss": loss_avg, "acc": acc_avg, "lr": lr, "char/s": chars, "GPU GB": gpu_memory_gb()}
        progress.step(metrics, write_every=100, step=step)
        if step % 100 == 0 and metrics:
            writer.add_scalar("train/loss", loss_avg, step)
            writer.add_scalar("train/masked_accuracy", acc_avg, step)
            writer.add_scalar("train/lr", lr, step)
        if step % args.eval_every == 0 or step == args.max_steps:
            valid_loss, accuracy = evaluate(model, valid_batches, device, autocast)
            progress.log(f"eval step {step}: validation loss {valid_loss:.3f} | "
                         f"masked-character accuracy {accuracy:.3f}\n{fill_examples(model, device)}")
            writer.add_scalar("valid/loss", valid_loss, step)
            writer.add_scalar("valid/accuracy", accuracy, step)
            with open(args.out / "eval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "valid_loss": valid_loss, "accuracy": accuracy}) + "\n")
        if time.time() - last_ckpt > args.ckpt_minutes * 60:
            save()
            last_ckpt = time.time()

    save()
    model.save_pretrained(args.out / "model")
    build_tokenizer().save_pretrained(args.out / "model")
    progress.log(f"Finished: Hugging Face model saved to {args.out / 'model'}")
    progress.close()


if __name__ == "__main__":
    main()
