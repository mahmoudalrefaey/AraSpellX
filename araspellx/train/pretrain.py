"""Pretrain AraSpellX's character encoder with masked character prediction.

Reads the token stream from data/pretrain_corpus.py, masks 15% of the
characters of each window (half as whole words, half as random 1-5
character spans) and trains BertForMaskedLM to restore them. The first
--short_fraction of the steps use 128-character windows, the rest 512.

Safety: every --cap_every steps, heads whose attention scores grow past
--attention_cap are scaled back (train/stability.py); the model with the
best validation loss is kept in best_model/ (and best.pt); training stops
by itself if validation loss gets --stop_worse worse than the best. A full
checkpoint is saved every --ckpt_minutes; rerunning the same command
resumes, and --resume_from starts a new output folder from another
checkpoint.

Short check run (~15 minutes on an RTX 3060 laptop GPU; validation
accuracy should climb past ~0.30), then the full run (~9 hours):

    python -m araspellx.train.pretrain --out artifacts/pretrain_check --max_steps 2000 --eval_every 500
    python -m araspellx.train.pretrain --out artifacts/pretrain --max_steps 80000
"""
from __future__ import annotations

import argparse
import json
import math
import sys
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
from araspellx.train.stability import cap_attention, max_attention_scores

WINDOW = 512  # the longest window: the model's positions
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
    """Random windows ([CLS] text [SEP]) from a uint8 token stream on disk."""

    def __init__(self, path: Path, seed: int) -> None:
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        self.rng = np.random.default_rng(seed)

    def batch(self, size: int, length: int = WINDOW):
        content = length - 2
        inputs, labels = [], []
        for _ in range(size):
            start = self.rng.integers(0, len(self.data) - content)
            chunk = np.asarray(self.data[start:start + content], dtype=np.int64)
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
    parser.add_argument("--max_steps", type=int, default=80000)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="characters per step, in 512-character windows (32: 16,384 characters)")
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden", type=int, default=384)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--ckpt_minutes", type=float, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                        help="auto: bf16 on GPUs that support it (RTX 30xx, A100), else fp16 (T4, V100)")
    parser.add_argument("--short_window", type=int, default=128,
                        help="window length for the first --short_fraction of the steps (0: full windows throughout)")
    parser.add_argument("--short_fraction", type=float, default=0.9)
    parser.add_argument("--attention_cap", type=float, default=50.0,
                        help="largest attention score allowed per head (0: no cap)")
    parser.add_argument("--cap_every", type=int, default=50, help="steps between attention-cap checks")
    parser.add_argument("--stop_worse", type=float, default=0.10,
                        help="stop when validation loss is this much worse than the best (0.10 = 10%%)")
    parser.add_argument("--resume_from", type=Path, default=None,
                        help="checkpoint to start from when --out has none yet (e.g. a copy of an earlier last.pt)")
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
    # Short windows first (as in the original BERT), then full windows; every step
    # sees the same number of characters.
    short_steps = int(args.short_fraction * args.max_steps) if args.short_window else 0

    def window_at(s: int) -> int:
        return args.short_window if s < short_steps else WINDOW

    def windows_per_batch(window: int) -> int:
        return args.batch_size * WINDOW // window

    train = Windows(args.data / "train.bin", args.seed)
    valid = Windows(args.data / "valid.bin", 1234)
    valid_batches = {w: [valid.batch(windows_per_batch(w), w) for _ in range(8)]
                     for w in sorted({window_at(0), WINDOW})}

    step, best = 0, {}  # best: window length -> (validation loss, step)
    ckpt = args.out / "last.pt"
    source = ckpt if ckpt.exists() else args.resume_from
    if source is not None:
        state = torch.load(source, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = state["step"]
        train.rng.bit_generator.state = state["data_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if scaler is not None and state.get("scaler"):
            scaler.load_state_dict(state["scaler"])
        best = state.get("best", {})
        del state

    progress = Progress(args.max_steps, step, args.out / "train.log", "pretrain")
    progress.log(f"Resumed from {source} at step {step}" if step else f"Starting at step 0 -> {args.out}")

    def save(path: Path, message: str):
        partial = path.with_suffix(".partial")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step,
                    "data_rng": train.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "scaler": scaler.state_dict() if scaler else None, "best": best, "args": vars(args)}, partial)
        partial.replace(path)
        progress.log(message)

    # A few validation windows (4,096 characters) on which attention scores are measured.
    probes = {w: batches[0][0][:4096 // w].to(device) for w, batches in valid_batches.items()}
    capped_heads, capped_recently = set(), {}

    def check_attention(window: int) -> None:
        for layer, head, score in cap_attention(model, probes[window], args.attention_cap):
            key = (layer, head)
            capped_recently[key] = max(capped_recently.get(key, 0.0), score)
            if key not in capped_heads:
                capped_heads.add(key)
                progress.log(f"attention cap: layer {layer} head {head} reached a score of {score:,.0f}; "
                             f"scaled back to {args.attention_cap:.0f}")

    params = sum(p.numel() for p in model.parameters())
    plan = (f"{args.short_window}-character windows until step {short_steps}, then {WINDOW}"
            if short_steps else f"{WINDOW}-character windows")
    progress.log(f"{params / 1e6:.1f}M parameters | {len(train.data) / 1e6:,.0f}M training characters | "
                 f"{args.max_steps} steps x {args.batch_size * WINDOW:,} characters | {plan} | "
                 f"{config.position_embedding_type} positions | precision {precision} | "
                 f"GPU {torch.cuda.get_device_name()}")
    writer = SummaryWriter(args.out / "logs")
    if args.attention_cap:
        check_attention(window_at(step))

    def batch_source(first_step: int):
        next_step = first_step  # batches are produced in step order

        def make():
            nonlocal next_step
            window = window_at(next_step)
            next_step += 1
            return train.batch(windows_per_batch(window), window)
        return make

    batches = Prefetcher(batch_source(step))
    last_ckpt = started = time.time()
    start_step, loss_avg, acc_avg, metrics, chars_seen = step, None, None, {}, 0
    while step < args.max_steps:
        if short_steps and step == short_steps:
            progress.log(f"switching to {WINDOW}-character windows")
        inputs, labels = batches.next()
        chars_seen += inputs.numel()
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
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        step += 1
        if args.attention_cap and step % args.cap_every == 0:
            check_attention(window_at(step))

        if step % 10 == 0:  # reading values back from the GPU every step would slow training
            masked = labels != -100
            accuracy = (logits[masked].argmax(-1) == labels[masked]).float().mean().item()
            loss_avg = loss.item() if loss_avg is None else 0.9 * loss_avg + 0.1 * loss.item()
            acc_avg = accuracy if acc_avg is None else 0.9 * acc_avg + 0.1 * accuracy
            chars = chars_seen / (time.time() - started)
            metrics = {"loss": loss_avg, "acc": acc_avg, "lr": lr, "grad": grad_norm.item(),
                       "char/s": chars, "GPU GB": gpu_memory_gb()}
        progress.step(metrics, write_every=100, step=step)
        if step % 100 == 0 and metrics:
            writer.add_scalar("train/loss", loss_avg, step)
            writer.add_scalar("train/masked_accuracy", acc_avg, step)
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/grad_norm", metrics["grad"], step)
        if step % args.eval_every == 0 or step == args.max_steps:
            window = window_at(step - 1)
            valid_loss, accuracy = evaluate(model, valid_batches[window], device, autocast)
            scores = max_attention_scores(model, probes[window]).amax(dim=1).tolist()  # per layer
            capped = ", ".join(f"layer {l} head {h} (up to {s:,.0f})" for (l, h), s in sorted(capped_recently.items()))
            capped_recently.clear()
            progress.log(f"eval step {step} ({window}-character windows): validation loss {valid_loss:.3f} | "
                         f"masked-character accuracy {accuracy:.3f}\n"
                         f"    largest attention score per layer: {' '.join(f'{s:.0f}' for s in scores)}"
                         f"{' | capped since last evaluation: ' + capped if capped else ''}\n"
                         f"{fill_examples(model, device)}")
            writer.add_scalar(f"valid{window}/loss", valid_loss, step)
            writer.add_scalar(f"valid{window}/accuracy", accuracy, step)
            for layer, score in enumerate(scores):
                writer.add_scalar(f"attention/max_score_layer{layer}", score, step)
            with open(args.out / "eval.jsonl", "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": step, "window": window, "valid_loss": valid_loss,
                                    "accuracy": accuracy, "max_attention_scores": scores}) + "\n")
            previous = best.get(window)
            if previous is None or valid_loss < previous[0]:
                best[window] = (valid_loss, step)
                save(args.out / "best.pt", f"new best validation loss for {window}-character windows: "
                                           f"{valid_loss:.3f} (saved best.pt and best_model)")
                model.save_pretrained(args.out / "best_model")
                build_tokenizer().save_pretrained(args.out / "best_model")
            elif valid_loss > previous[0] * (1 + args.stop_worse):
                progress.log(f"STOPPED: validation loss {valid_loss:.3f} is more than {args.stop_worse:.0%} worse "
                             f"than the best ({previous[0]:.3f} at step {previous[1]}). The best model is in "
                             f"{args.out / 'best_model'} and its full checkpoint in {args.out / 'best.pt'}.")
                progress.close()
                sys.exit(1)
        if time.time() - last_ckpt > args.ckpt_minutes * 60:
            save(ckpt, f"checkpoint saved at step {step}")
            last_ckpt = time.time()

    save(ckpt, f"checkpoint saved at step {step}")
    model.save_pretrained(args.out / "model")
    build_tokenizer().save_pretrained(args.out / "model")
    progress.log(f"Finished: Hugging Face model saved to {args.out / 'model'} "
                 f"(best validation: {', '.join(f'{w} characters {l:.3f} at step {s}' for w, (l, s) in sorted(best.items()))})")
    progress.close()


if __name__ == "__main__":
    main()
