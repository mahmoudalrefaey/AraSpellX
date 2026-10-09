"""Evaluate a correction model against the release gates (docs/evaluation.md).

The model reads every development and test text once; each confidence
threshold is then applied to the stored predictions. The threshold is chosen
on development data only: the highest mean F0.5 over the development error
sets among thresholds whose damage on clean development text stays within
--damage_budget. Every frozen test set is then scored at that threshold and
checked against the gates. Writes report.md and results.json to --out.

    python -m araspellx.eval.run --model artifacts/correct/best_model --out artifacts/eval/correct
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import string
import time
from bisect import bisect_left
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rapidfuzz.distance import Levenshtein

from araspellx.correct.decode import Prediction, at_threshold, predict
from araspellx.data.labels import KEEP, LabelVocab, allowed_label, apply_labels, derive_labels
from araspellx.eval.metrics import Score, combine, regions_in, score, word_regions
from araspellx.model.bert import BertForTokenClassification
from araspellx.noise.typed import typed_noise
from araspellx.text.normalize import normalize
from araspellx.train.correct import load_split_pairs
from araspellx.train.correction_data import Paragraphs

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
DEV_ERROR_SETS = ("dev_typed", "dev_ocr", "dev_yarmouk")
TEST_ERROR_SETS = ("T1_typed", "T4_tesseract", "T4_abbyy", "T6_tesseract", "T6_openiti", "benchmark")
DESCRIPTIONS = {
    "dev_clean": "clean held-out paragraphs (damage)",
    "dev_typed": "held-out paragraphs with typed-error noise",
    "dev_ocr": "development share of the OCR training pairs",
    "dev_yarmouk": "Yarmouk development split, real scans",
    "T1_typed": "T-1 real typed errors: paragraphs before and after spelling fixes in Wikipedia's edit history",
    "T4_tesseract": "T-4 real scans, Tesseract",
    "T4_abbyy": "T-4 real scans, ABBYY",
    "T6_tesseract": "T-6 classical books, Tesseract",
    "T6_openiti": "T-6 classical books, OpenITI OCR",
    "T7_msa": "T-7 clean modern text (damage)",
    "T7_classical": "T-7 clean classical text (damage)",
    "T7_diacritized": "T-7 clean diacritized text (damage)",
    "T7_dialect": "T-7 clean dialect text (damage)",
    "benchmark": "27 hand-corrected typed sentences",
}

Item = Tuple[str, str, str]  # (source, reference, group: the page a paragraph belongs to, or "")


def _norm(text: str) -> str:
    return normalize(text).text


def _jsonl(path: Path) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_sets(args) -> Dict[str, List[Item]]:
    """Development sets (for the threshold) and frozen test sets, normalized like inference input."""
    rng = random.Random(1234)
    valid = Paragraphs(args.text / "valid.bin")
    clean = [valid.sample(rng) for _ in range(args.dev_clean)]
    typed = [valid.sample(rng) for _ in range(args.dev_typed)]
    sets = {"dev_clean": [(c, c, "") for c in clean],
            "dev_typed": [(typed_noise(c, rng), c, "") for c in typed]}
    _, ocr_dev = load_split_pairs([p for p in args.ocr_pairs if p.exists()])
    sets["dev_ocr"] = [(n, c, "") for n, c in ocr_dev]
    sets["dev_yarmouk"] = [(_norm(r["noisy"]), _norm(r["clean"]), "")
                           for r in _jsonl(args.testsets / "yarmouk_dev.jsonl")]
    if (args.testsets / "T1_wiki_edits.jsonl").exists():
        sets["T1_typed"] = [(_norm(r["noisy"]), _norm(r["clean"]), "")
                            for r in _jsonl(args.testsets / "T1_wiki_edits.jsonl")]
    t4 = _jsonl(args.testsets / "T4_yarmouk_test.jsonl")
    for engine in ("tesseract", "abbyy"):
        sets[f"T4_{engine}"] = [(_norm(r["noisy"]), _norm(r["clean"]), r["id"].rsplit(":", 1)[0])
                                for r in t4 if r["engine"] == engine]
    t6 = _jsonl(args.testsets / "T6_openiti.jsonl")
    for engine in ("tesseract", "openiti"):
        sets[f"T6_{engine}"] = [(_norm(r["noisy"]), _norm(r["clean"]), "") for r in t6 if r["engine"] == engine]
    for kind in ("msa", "classical", "diacritized", "dialect"):
        sets[f"T7_{kind}"] = [(_norm(r["text"]), _norm(r["text"]), "")
                              for r in _jsonl(args.testsets / f"T7_{kind}.jsonl")]
    with open(args.benchmark, encoding="utf-8") as f:
        sets["benchmark"] = [(_norm(r["distorted"]), _norm(r["clean"]), "") for r in csv.DictReader(f)]
    if args.limit:
        sets = {name: items[:args.limit] for name, items in sets.items()}
    return sets


def outputs_at(items: Sequence[Item], predictions: Sequence[Prediction], threshold: float) -> List[str]:
    return [apply_labels(src, at_threshold(p, src, threshold)) for (src, _, _), p in zip(items, predictions)]


def _score_pair(job) -> List[Score]:
    """Scores of one text at every threshold; identical outputs are scored once."""
    src, ref, outputs = job
    cache = {}
    for out in outputs:
        if out not in cache:
            cache[out] = score([src], [out], [ref], category=None)
    return [cache[out] for out in outputs]


def _score_with_categories(job) -> Score:
    src, ref, out = job
    return score([src], [out], [ref])


def score_all(pool, items: Sequence[Item], predictions: Sequence[Prediction],
              thresholds: Sequence[float]) -> Dict[float, Score]:
    """Scores of a set at each threshold (texts are scored in parallel)."""
    jobs = [(src, ref, [apply_labels(src, at_threshold(p, src, t)) for t in thresholds])
            for (src, ref, _), p in zip(items, predictions)]
    per_text = pool.map(_score_pair, jobs, chunksize=8)
    return {t: combine([scores[k] for scores in per_text]) for k, t in enumerate(thresholds)}


def pages_no_worse(items: Sequence[Item], outputs: Sequence[str]) -> Tuple[int, int]:
    """(pages whose character errors did not increase, pages)."""
    errors: Dict[str, List[int]] = {}
    for (src, ref, page), out in zip(items, outputs):
        before_after = errors.setdefault(page, [0, 0])
        before_after[0] += Levenshtein.distance(src, ref)
        before_after[1] += Levenshtein.distance(out, ref)
    return sum(after <= before for before, after in errors.values()), len(errors)


def _proposals(job) -> List[Tuple[float, bool]]:
    """(confidence, correct) of every edit the model proposes in one text.

    A proposal is the allowed part of a character's most likely label when it
    is not "keep"; it is correct if it equals the label that turns the source
    into the reference.
    """
    src, ref, prediction = job
    gold = derive_labels(src, ref).chars
    result = []
    for i, (label, prob) in enumerate(zip(prediction.chars, prediction.probs)):
        label = allowed_label(src, i, label)
        if label != KEEP:
            result.append((prob, label == gold[i]))
    return result


def proposals(pool, items: Sequence[Item], predictions: Sequence[Prediction]) -> List[Tuple[float, bool]]:
    jobs = [(src, ref, p) for (src, ref, _), p in zip(items, predictions)]
    return [pair for pairs in pool.map(_proposals, jobs, chunksize=8) for pair in pairs]


def fit_isotonic(pairs: Sequence[Tuple[float, bool]]) -> List[Tuple[float, float]]:
    """Monotone map from confidence to observed accuracy (pool adjacent violators).

    Returns (highest confidence of the block, accuracy of the block) in order.
    """
    blocks = []  # [correct, count, highest confidence]
    for prob, correct in sorted(pairs):
        blocks.append([float(correct), 1, prob])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            correct_sum, count, top = blocks.pop()
            blocks[-1][0] += correct_sum
            blocks[-1][1] += count
            blocks[-1][2] = top
    return [(top, correct_sum / count) for correct_sum, count, top in blocks]


def calibrate(prob: float, mapping: Sequence[Tuple[float, float]]) -> float:
    index = bisect_left([top for top, _ in mapping], prob)
    return mapping[min(index, len(mapping) - 1)][1]


def calibration_table(pairs: Sequence[Tuple[float, bool]], bins: int = 10):
    """Expected calibration error and a reliability table of (confidence, correct) pairs."""
    table = [[0, 0, 0.0] for _ in range(bins)]  # edits, correct, summed confidence
    for prob, correct in pairs:
        row = table[min(int(prob * bins), bins - 1)]
        row[0] += 1
        row[1] += correct
        row[2] += prob
    total = sum(row[0] for row in table)
    ece = sum(abs(row[1] - row[2]) for row in table) / total if total else 0.0
    rows = [{"from": b / bins, "to": (b + 1) / bins, "edits": row[0],
             "accuracy": row[1] / row[0] if row[0] else None,
             "confidence": row[2] / row[0] if row[0] else None} for b, row in enumerate(table)]
    return ece, total, rows


PUNCTUATION = " " + string.punctuation + "،؛؟«»…"


def _targeted(job):
    """T-1 counts for one paragraph: the words an editor fixed and what the model did to them."""
    src, ref, out = job
    regions = word_regions(src)
    fixed = touched = exact = 0
    others = []  # edits of words the editor left alone (often errors the editor did not fix)
    for (a, b), r, o in zip(regions, regions_in(src, ref, regions), regions_in(src, out, regions)):
        s = src[a:b]
        if r != s:
            fixed += 1
            if o != s:
                touched += 1
                exact += o == r
        elif o != s:
            others.append((s.strip().strip(PUNCTUATION), o.strip().strip(PUNCTUATION)))
    return fixed, touched, exact, len(regions), others


def known_fixes(path: Path) -> set:
    """(before, after) word fixes that Wikipedia editors made on training pages."""
    fixes = set()
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                fixes.update(tuple(fix) for fix in json.loads(line)["fixes"])
    return fixes


def targeted(pool, items: Sequence[Item], outputs: Sequence[str], fixes: set) -> Dict:
    """T-1 scored on the words the editors fixed: a real edit fixes only some of a paragraph's
    errors, so the model's other edits cannot be scored against it; they are counted, and
    checked against the fixes editors made on other pages."""
    fixed = touched = exact = words = 0
    others = []
    for f, t, e, w, o in pool.map(_targeted, [(s, r, out) for (s, r, _), out in zip(items, outputs)], chunksize=8):
        fixed, touched, exact, words = fixed + f, touched + t, exact + e, words + w
        others += o
    attested = [pair in fixes for pair in others]
    samples = random.Random(0).sample(range(len(others)), min(16, len(others)))
    return {"fixed": fixed, "touched": touched, "exact": exact, "recall": exact / fixed if fixed else 0.0,
            "precision": exact / touched if touched else 1.0, "words": words, "other_edits": len(others),
            "other_per_1000_words": 1000 * len(others) / words if words else 0.0,
            "other_attested": sum(attested) / len(others) if others else 0.0,
            "samples": [(*others[i], attested[i]) for i in samples]}


def harmful_examples(items: Sequence[Item], outputs: Sequence[str], count: int, seed: int = 0):
    """Edits that changed a correct word or did not bring a wrong word closer: (input, output, reference)."""
    found = []
    for (src, ref, _), out in zip(items, outputs):
        regions = word_regions(src)
        for (a, b), r, o in zip(regions, regions_in(src, ref, regions), regions_in(src, out, regions)):
            s = src[a:b]
            if o != s and (r == s or (o != r and Levenshtein.distance(o, r) >= Levenshtein.distance(s, r))):
                found.append((s.strip(), o.strip(), r.strip()))
    return random.Random(seed).sample(found, min(count, len(found)))


def choose_threshold(dev: Dict[str, Dict[float, Score]], budget: float) -> Tuple[float, str]:
    allowed = [t for t in THRESHOLDS if dev["dev_clean"][t].damage <= budget]
    if not allowed:
        return THRESHOLDS[-1], f"no threshold kept clean-text damage within {budget:.2%}; using the highest"
    mean_f05 = {t: sum(dev[name][t].f05 for name in DEV_ERROR_SETS) / len(DEV_ERROR_SETS) for t in allowed}
    best = max(allowed, key=lambda t: mean_f05[t])
    return best, (f"highest mean F0.5 over the development error sets ({mean_f05[best]:.3f}) among thresholds "
                  f"keeping damage on clean development text within {budget:.2%}")


def _relative_drop(before: float, after: float) -> float:
    return (before - after) / before if before else 0.0


def gates(test: Dict[str, Score], pages: Tuple[int, int], ece: Optional[float], raw_ece: Optional[float],
          t1: Optional[Dict]) -> List[Dict]:
    t4 = combine([test["T4_tesseract"], test["T4_abbyy"]])
    t6 = combine([test["T6_tesseract"], test["T6_openiti"]])
    t4_rates, t6_rates = t4.rates(), t6.rates()
    wer_drop = _relative_drop(t4_rates["wer_in"], t4_rates["wer_out"])

    def gate(name, target, value, passed):
        return {"gate": name, "target": target, "measured": value,
                "result": "not measured" if passed is None else ("PASS" if passed else "FAIL")}

    return [
        gate("Damage, clean modern text (T-7)", "≤ 0.1% of words",
             f"{test['T7_msa'].damage:.3%}", test["T7_msa"].damage <= 0.001),
        gate("Damage, clean classical text (T-7)", "≤ 0.2%",
             f"{test['T7_classical'].damage:.3%}", test["T7_classical"].damage <= 0.002),
        gate("Damage, clean diacritized text (T-7)", "≤ 0.2%",
             f"{test['T7_diacritized'].damage:.3%}", test["T7_diacritized"].damage <= 0.002),
        gate("Damage, dialect text (T-7)", "≤ 0.5%",
             f"{test['T7_dialect'].damage:.3%}", test["T7_dialect"].damage <= 0.005),
        gate("Real typed errors (T-1): edit precision", "≥ 0.90 on the words editors fixed",
             f"{t1['precision']:.3f} (recall {t1['recall']:.3f}); elsewhere {t1['other_per_1000_words']:.1f} edits "
             f"per 1,000 words, {t1['other_attested']:.0%} of them fixes editors made on other pages"
             if t1 else "T-1 not built yet", t1["precision"] >= 0.90 if t1 else None),
        gate("Real OCR, modern (T-4): word errors removed", "≥ 20%",
             f"{wer_drop:.1%} (WER {t4_rates['wer_in']:.1%} → {t4_rates['wer_out']:.1%})", wer_drop >= 0.20),
        gate("Real OCR, modern (T-4): harmful edits", "≤ 5% of the model's edits",
             f"{t4.harmful:.1%} (exact-fix precision {t4.precision:.3f})", t4.harmful <= 0.05),
        gate("Real OCR, modern (T-4): pages no worse than raw OCR", "≥ 98%",
             f"{pages[0] / pages[1]:.1%} ({pages[0]}/{pages[1]})", pages[0] / pages[1] >= 0.98),
        gate("Real OCR, classical (T-6): no harm", "error rates not higher",
             f"CER {t6_rates['cer_in']:.1%} → {t6_rates['cer_out']:.1%}, WER {t6_rates['wer_in']:.1%} → "
             f"{t6_rates['wer_out']:.1%}",
             t6_rates["cer_out"] <= t6_rates["cer_in"] and t6_rates["wer_out"] <= t6_rates["wer_in"]),
        gate("Degraded scans (T-5): no harm", "error rates not higher", "T-5 not built yet", None),
        gate("Speed", "≥ 300 words/s, 4-core laptop CPU", "measured by araspellx.eval.speed", None),
        gate("Long documents", "within 10% of sentence quality", "not measured yet", None),
        gate("Confidence calibration (T-4 + T-6)", "ECE ≤ 0.05",
             "no edits" if ece is None else f"{ece:.3f} after calibration on development OCR data (raw {raw_ece:.3f})",
             None if ece is None else ece <= 0.05),
    ]


def _row(name: str, items: Sequence[Item], s: Score) -> str:
    r = s.rates()
    words = sum(len(src.split()) for src, _, _ in items)
    if name.startswith(("T7", "dev_clean")):
        return (f"| {name} | {len(items):,} | {words:,} | – | – | – | – | **{s.damage:.3%}** | "
                f"{r['cer_in']:.2%} → {r['cer_out']:.2%} | {r['wer_in']:.1%} → {r['wer_out']:.1%} |")
    return (f"| {name} | {len(items):,} | {words:,} | {s.precision:.3f} | {s.recall:.3f} | {s.f05:.3f} | "
            f"{s.harmful:.1%} | {s.damage:.2%} | {r['cer_in']:.1%} → {r['cer_out']:.1%} | "
            f"{r['wer_in']:.1%} → {r['wer_out']:.1%} |")


def write_report(path: Path, model: Path, threshold: float, reason: str, sets: Dict[str, List[Item]],
                 scores: Dict[str, Dict[float, Score]], gate_rows: List[Dict], ece_rows, ece_edits: int, raw_ece: float,
                 categories: Score, examples: Dict[str, list], seconds: float, t1: Optional[Dict]) -> None:
    header = ("| Set | Texts | Words | Precision | Recall | F0.5 | Harmful edits | Damage | CER in → out | "
              "WER in → out |\n|---|---|---|---|---|---|---|---|---|---|")
    lines = [f"# Evaluation of `{model}`", "",
             f"{time.strftime('%Y-%m-%d %H:%M')} · {seconds / 60:.0f} minutes · threshold **{threshold}**: {reason}.",
             "", "Precision counts exact fixes only; *harmful edits* is the share of the model's edits that changed "
             "a correct word or did not bring a wrong word closer to the reference; *damage* is the share of "
             "correct words the model changed. References are not perfect: some edits counted as harmful fix "
             "errors in the reference itself.", "", "## Release gates", "",
             "| Gate | Target | Measured | Result |", "|---|---|---|---|"]
    lines += [f"| {g['gate']} | {g['target']} | {g['measured']} | {g['result']} |" for g in gate_rows]
    lines += ["", f"## Test sets at threshold {threshold}", "", header]
    lines += [_row(name, sets[name], scores[name][threshold]) for name in sets if not name.startswith("dev")]
    if t1:
        lines += ["", f"## Real typed errors (T-1) at threshold {threshold}", "",
                  "Each paragraph records the fixes of one Wikipedia edit; the paragraph's other errors stay in the "
                  "reference, so the strict scores in the table above count the model's fixes of them as damage. "
                  "T-1 is therefore scored on the words the editors fixed, and the model's other edits are counted "
                  "and checked against fixes that editors made on other pages.", "",
                  f"- Words fixed by the editors: {t1['fixed']:,}; the model fixed {t1['exact']:,} of them exactly "
                  f"(recall **{t1['recall']:.3f}**)",
                  f"- Of the {t1['touched']:,} of these words the model changed, {t1['exact']:,} match the editor "
                  f"(precision **{t1['precision']:.3f}**)",
                  f"- Other edits: {t1['other_edits']:,} ({t1['other_per_1000_words']:.1f} per 1,000 words); "
                  f"**{t1['other_attested']:.0%}** are fixes that editors made on other pages", "",
                  "Samples of other edits (✓: a fix editors made elsewhere):", ""]
        lines += [f"- {a} → {b} {'✓' if ok else ''}" for a, b, ok in t1["samples"]]
    lines += ["", "## Threshold choice (development data only)", "",
              "| Threshold | Damage on clean text | " + " | ".join(f"{n} P / R" for n in DEV_ERROR_SETS) + " |",
              "|---|---|" + "---|" * len(DEV_ERROR_SETS)]
    for t in THRESHOLDS:
        cells = " | ".join(f"{scores[n][t].precision:.2f} / {scores[n][t].recall:.2f}" for n in DEV_ERROR_SETS)
        mark = " ◀" if t == threshold else ""
        lines.append(f"| {t}{mark} | {scores['dev_clean'][t].damage:.3%} | {cells} |")
    lines += ["", "## Test sets at every threshold (precision / recall / damage)", "",
              "| Set | " + " | ".join(str(t) for t in THRESHOLDS) + " |", "|---|" + "---|" * len(THRESHOLDS)]
    for name in sets:
        if not name.startswith("dev"):
            lines.append(f"| {name} | " + " | ".join(
                f"{scores[name][t].precision:.2f} / {scores[name][t].recall:.2f} / {scores[name][t].damage:.2%}"
                for t in THRESHOLDS) + " |")
    lines += ["", "## Error categories, real scans (T-4, both engines)", "",
              "| Category | Errors | Fixed exactly | Model edits | Exact |", "|---|---|---|---|---|"]
    for category, c in sorted(categories.by_category.items(), key=lambda kv: -kv[1]["gold"]):
        lines.append(f"| {category} | {c['gold']} | {c['correct'] / c['gold']:.0%} | {c['system']} | "
                     f"{c['correct'] / c['system']:.0%} |" if c["gold"] and c["system"] else
                     f"| {category} | {c['gold']} | – | {c['system']} | – |")
    lines += ["", f"## Confidence calibration (T-4 + T-6, {ece_edits:,} proposed character edits)", "",
              f"Confidences mapped by an isotonic fit on the development OCR sets (`calibration.json` also "
              f"holds a fit on typed text); "
              f"expected calibration error {raw_ece:.3f} before the mapping.", "",
              "| Confidence | Edits | Mean confidence | Accuracy |", "|---|---|---|---|"]
    lines += [f"| {r['from']:.1f}–{r['to']:.1f} | {r['edits']:,} | {r['confidence']:.2f} | {r['accuracy']:.2f} |"
              for r in ece_rows if r["edits"]]
    lines += ["", f"## Harmful edits at threshold {threshold} (samples: input → output ; reference)", ""]
    for name, rows in examples.items():
        lines.append(f"**{name}**")
        lines += [f"- {s} → {o} ; {r or '(deleted)'}" for s, o, r in rows] or ["- none"]
        lines.append("")
    lines += ["## Sets", ""] + [f"- **{name}**: {DESCRIPTIONS[name]}" for name in sets]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("artifacts/correct/best_model"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/correct"))
    parser.add_argument("--text", type=Path, default=Path("data/v1/pretrain"))
    parser.add_argument("--testsets", type=Path, default=Path("data/v1/testsets"))
    parser.add_argument("--ocr_pairs", type=Path, nargs="+",
                        default=[Path("data/v1/pairs/ocr_render.jsonl"), Path("data/v1/pairs/yarmouk_train.jsonl")])
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/real_world.csv"))
    parser.add_argument("--edits_train", type=Path, default=Path("data/v1/pairs/wiki_edits_train.jsonl"),
                        help="mined training edits: their fixes check the model's other edits on T-1")
    parser.add_argument("--dev_clean", type=int, default=2000, help="clean development paragraphs")
    parser.add_argument("--dev_typed", type=int, default=1000, help="development paragraphs with typed noise")
    parser.add_argument("--damage_budget", type=float, default=0.0005,
                        help="largest damage on clean development text allowed for the threshold (0.0005 = 0.05%%)")
    parser.add_argument("--threshold", type=float, default=None, help="use this threshold instead of choosing one")
    parser.add_argument("--limit", type=int, default=0, help="texts per set (0: all), for quick checks")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=min(8, cpu_count()), help="processes for scoring")
    args = parser.parse_args()

    started = time.time()
    device = torch.device(args.device)
    model = BertForTokenClassification.from_pretrained(args.model).to(device).eval()
    vocab = LabelVocab.load(args.model / "labels.json")
    sets = load_sets(args)
    predictions = {}
    for name, items in sets.items():
        t = time.time()
        predictions[name] = predict(model, [src for src, _, _ in items], vocab, device)
        print(f"{name:15s} {len(items):6,} texts read in {time.time() - t:5.0f} s", flush=True)

    pool = Pool(args.workers)
    t = time.time()
    scores = {name: score_all(pool, items, predictions[name], THRESHOLDS) for name, items in sets.items()}
    print(f"scored {len(THRESHOLDS)} thresholds in {time.time() - t:.0f} s", flush=True)
    threshold, reason = (args.threshold, "given on the command line") if args.threshold is not None else \
        choose_threshold({n: scores[n] for n in scores if n.startswith("dev")}, args.damage_budget)
    if threshold not in THRESHOLDS:
        for name, items in sets.items():
            scores[name].update(score_all(pool, items, predictions[name], [threshold]))
    print(f"threshold {threshold}: {reason}", flush=True)

    outputs = {name: outputs_at(items, predictions[name], threshold) for name, items in sets.items()}
    test = {name: scores[name][threshold] for name in sets if not name.startswith("dev")}
    t4_items = sets["T4_tesseract"] + sets["T4_abbyy"]
    t4_outputs = outputs["T4_tesseract"] + outputs["T4_abbyy"]
    pages = pages_no_worse([(s, r, f"{i < len(sets['T4_tesseract'])}:{g}") for i, (s, r, g) in enumerate(t4_items)],
                           t4_outputs)
    categories = combine(pool.map(_score_with_categories,
                                  [(s, r, o) for (s, r, _), o in zip(t4_items, t4_outputs)], chunksize=8))
    error_items = [item for name in ("T4_tesseract", "T4_abbyy", "T6_tesseract", "T6_openiti") for item in sets[name]]
    error_predictions = [p for name in ("T4_tesseract", "T4_abbyy", "T6_tesseract", "T6_openiti")
                         for p in predictions[name]]
    # One confidence map per kind of input, fitted on development data: pipelines know whether
    # their text comes from OCR or was typed. Real-scan test sets are checked with the OCR map.
    mappings = {kind: fit_isotonic(proposals(pool, [i for n in names for i in sets[n]],
                                             [p for n in names for p in predictions[n]]))
                for kind, names in (("ocr", ("dev_ocr", "dev_yarmouk")), ("typed", ("dev_typed",)))}
    test_pairs = proposals(pool, error_items, error_predictions)
    raw_ece, ece_edits, _ = calibration_table(test_pairs)
    ece, _, ece_rows = calibration_table([(calibrate(prob, mappings["ocr"]), ok) for prob, ok in test_pairs])
    t1 = None
    if "T1_typed" in sets:
        t1 = targeted(pool, sets["T1_typed"], outputs["T1_typed"], known_fixes(args.edits_train))
    pool.close()
    gate_rows = gates(test, pages, ece if ece_edits else None, raw_ece, t1)
    examples = {name: harmful_examples(sets[name], outputs[name], 8)
                for name in TEST_ERROR_SETS + ("T7_msa",) if name in sets}

    args.out.mkdir(parents=True, exist_ok=True)
    write_report(args.out / "report.md", args.model, threshold, reason, sets, scores, gate_rows, ece_rows,
                 ece_edits, raw_ece, categories, examples, time.time() - started, t1)
    (args.out / "calibration.json").write_text(json.dumps(
        {"method": "isotonic fit of edit confidence to accuracy on development data, per kind of input",
         "threshold": threshold,
         **{kind: [{"up_to": top, "probability": value} for top, value in mapping] for kind, mapping in mappings.items()}},
        indent=1), encoding="utf-8")
    results = {"model": str(args.model), "threshold": threshold, "reason": reason, "gates": gate_rows,
               "pages_no_worse": pages, "ece": ece, "raw_ece": raw_ece, "calibration": ece_rows,
               "scores": {name: {str(t): s.summary() for t, s in by_t.items()} for name, by_t in scores.items()},
               "t4_categories": {k: dict(v) for k, v in categories.by_category.items()}, "t1": t1}
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(f"{g['result']:>12s}  {g['gate']}: {g['measured']} (target {g['target']})" for g in gate_rows))
    print(f"report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
