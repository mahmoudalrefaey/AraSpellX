"""Controlled typo/habit probes for comparing checkpoints like-for-like.

Each probe takes a clean test sentence, corrupts it in one controlled way
(one or more typos in chosen words, or the spelling habits), and checks
whether the model restores the clean sentence. Probe sentences depend only
on --seed and the source CSV, never on the checkpoint.

    python evaluate_probes.py --checkpoint outdir/checkpoint_epoch_0_step_35000.pt
"""
import argparse
import csv
import random

import editdistance
import torch

from core import constants
from data.processes import RandomNeighborReplacer
from data.processors import get_habits
from inference import DEFAULT_TOKENIZER, correct_texts, load_model


KEYBOARD = RandomNeighborReplacer(
    constants.KEYBOARD_KEYS, constants.KEYBOARD_BLANK
)._mapper

# Severity probes: (number of corrupted words, typos per word, apply habits).
SEVERITY = {
    "1 typo": (1, 1, False),
    "2 typos in one word": (1, 2, False),
    "3 typos in one word": (1, 3, False),
    "1 typo in each of 3 words": (3, 1, False),
    "habits only": (1, 0, True),
    "habits + 2 typos in one word": (1, 2, True),
}

SINGLE_TYPOS = [
    "deleted letter",
    "doubled letter",
    "keyboard substitution",
    "swapped letters",
    "inserted letter",
]


def typo(word, kind, rng):
    """Apply one typo of the given kind to word, or return None if it can't."""
    i = rng.randrange(len(word))
    if kind == "deleted letter":
        return word[:i] + word[i + 1:] if len(word) > 2 else None
    if kind == "doubled letter":
        return word[:i] + word[i] + word[i:]
    if kind == "keyboard substitution":
        if not KEYBOARD.get(word[i]):
            return None
        return word[:i] + rng.choice(KEYBOARD[word[i]]) + word[i + 1:]
    if kind == "swapped letters":
        if i == len(word) - 1 or word[i] == word[i + 1]:
            return None
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word[:i] + rng.choice(constants.ARABIC_CHARS) + word[i:]


def random_typos(word, n, rng):
    for _ in range(n):
        corrupted = None
        while corrupted is None:
            corrupted = typo(word, rng.choice(SINGLE_TYPOS), rng)
        word = corrupted
    return word


def apply_habits(sentence, habits):
    for habit in habits:
        sentence = habit.execute(sentence)
    return sentence


def build_probes(sentences, seed, n_severity, n_single):
    """Return {category: [(clean, corrupted, target word indices)]}."""
    habits = get_habits(apply_prob=1.0)
    probes = {}

    for category, (n_words, n_typos, use_habits) in SEVERITY.items():
        rng = random.Random(f"{seed}-{category}")
        items = []
        for clean in rng.sample(sentences, len(sentences)):
            words = clean.split(" ")
            candidates = [i for i, w in enumerate(words) if len(w) >= 4]
            if len(candidates) < n_words:
                continue
            targets = rng.sample(candidates, n_words)
            corrupted = apply_habits(clean, habits).split(" ") if use_habits else list(words)
            for t in targets:
                corrupted[t] = random_typos(corrupted[t], n_typos, rng)
            corrupted = " ".join(corrupted)
            if corrupted != clean:
                items.append((clean, corrupted, targets))
            if len(items) == n_severity:
                break
        probes[category] = items

    for kind in SINGLE_TYPOS:
        rng = random.Random(f"{seed}-{kind}")
        items = []
        for clean in rng.sample(sentences, len(sentences)):
            words = clean.split(" ")
            candidates = [i for i, w in enumerate(words) if len(w) >= 4]
            if not candidates:
                continue
            t = rng.choice(candidates)
            corrupted_word = typo(words[t], kind, rng)
            if corrupted_word is None or corrupted_word == words[t]:
                continue
            corrupted = list(words)
            corrupted[t] = corrupted_word
            items.append((clean, " ".join(corrupted), [t]))
            if len(items) == n_single:
                break
        probes[f"single: {kind}"] = items

    return probes


def score(items, outputs):
    counts = {"fixed": 0, "left": 0, "wrong": 0, "word count": 0, "exact": 0}
    damaged = other = 0

    for (clean, corrupted, targets), output in zip(items, outputs):
        counts["exact"] += output == clean
        clean_words = clean.split(" ")
        input_words = corrupted.split(" ")
        output_words = output.split(" ")

        if len(output_words) != len(clean_words):
            counts["word count"] += 1
            continue

        if all(output_words[t] == clean_words[t] for t in targets):
            counts["fixed"] += 1
        elif all(output_words[t] == input_words[t] for t in targets):
            counts["left"] += 1
        else:
            counts["wrong"] += 1

        rest = [i for i in range(len(clean_words)) if i not in targets]
        damaged += sum(output_words[i] != clean_words[i] for i in rest)
        other += len(rest)

    n = len(items)
    return {
        "n": n,
        "fixed": counts["fixed"] / n,
        "left": counts["left"] / n,
        "wrong": counts["wrong"] / n,
        "word count": counts["word count"] / n,
        "exact": counts["exact"] / n,
        "damaged": damaged / max(other, 1),
    }


def error_rates(hypotheses, references):
    chars = sum(len(r) for r in references)
    words = sum(len(r.split()) for r in references)
    cer = sum(editdistance.eval(h, r) for h, r in zip(hypotheses, references)) / chars
    wer = sum(
        editdistance.eval(h.split(), r.split())
        for h, r in zip(hypotheses, references)
    ) / words
    exact = sum(h == r for h, r in zip(hypotheses, references)) / len(references)
    return cer, wer, exact


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER, type=str)
    parser.add_argument(
        "--source_csv", default="data/dataset/test.csv", type=str,
        help="CSV whose clean column provides the probe sentences",
    )
    parser.add_argument("--clean_col", default="clean", type=str)
    parser.add_argument(
        "--source_rows", default=20000, type=int,
        help="Number of source rows the probes are drawn from",
    )
    parser.add_argument("--n_severity", default=150, type=int)
    parser.add_argument("--n_single", default=120, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--benchmark", default="data/benchmarks/real_world.csv", type=str,
        help="Real-world CSV (clean, distorted) to score as well; '' to skip",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.checkpoint, args.tokenizer, device)

    with open(args.source_csv, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        sentences = [
            row[args.clean_col] for _, row in zip(range(args.source_rows), reader)
        ]

    probes = build_probes(
        sentences, args.seed, args.n_severity, args.n_single
    )

    print()
    print(f"Probes for {args.checkpoint}")
    print(
        f"{'probe':34s} {'n':>4s} {'fixed':>6s} {'left':>6s} {'wrong':>6s} "
        f"{'words±':>6s} {'exact':>6s} {'damage':>7s}"
    )
    for category, items in probes.items():
        outputs = correct_texts(
            model, tokenizer, [corrupted for _, corrupted, _ in items], device
        )
        s = score(items, outputs)
        print(
            f"{category:34s} {s['n']:4d} {s['fixed']:6.3f} {s['left']:6.3f} "
            f"{s['wrong']:6.3f} {s['word count']:6.3f} {s['exact']:6.3f} "
            f"{s['damaged']:7.4f}"
        )
    print(
        "fixed/left/wrong: corrupted word(s) restored / copied unchanged / "
        "changed to something else; words±: word count changed; "
        "damage: share of the other, correct words that were changed"
    )

    if args.benchmark:
        with open(args.benchmark, encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        inputs = [row["distorted"] for row in rows]
        references = [row["clean"] for row in rows]
        outputs = correct_texts(model, tokenizer, inputs, device)
        print()
        print(f"Benchmark {args.benchmark} ({len(rows)} rows)")
        print("Input (no correction): CER=%.4f WER=%.4f exact=%.3f" % error_rates(inputs, references))
        print("Model output:          CER=%.4f WER=%.4f exact=%.3f" % error_rates(outputs, references))


if __name__ == "__main__":
    main()
