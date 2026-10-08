"""Evaluate a correction model against the release gates (docs/evaluation.md).

The model reads every development and test text once; each confidence
threshold is then applied to the stored predictions. The threshold is chosen
on development data only: the highest mean F0.5 over the development error
sets among thresholds whose damage on clean development text stays within
--damage_budget. Every frozen test set is then scored at that threshold and
checked against the gates. Writes report.md and results.json to --out.

    python -m araspellx.eval.run --model artifacts/correct/model --out artifacts/eval/correct
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from rapidfuzz.distance import Levenshtein

from araspellx.correct.decode import Prediction, at_threshold, predict
from araspellx.data.labels import KEEP, LabelVocab, apply_labels, derive_labels
from araspellx.eval.metrics import Score, combine, regions_in, score, word_regions
from araspellx.model.bert import BertForTokenClassification
from araspellx.noise.typed import typed_noise
from araspellx.text.charset import is_editable
from araspellx.text.normalize import normalize
from araspellx.train.correct import load_split_pairs
from araspellx.train.correction_data import Paragraphs

THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.98)
DEV_ERROR_SETS = ("dev_typed", "dev_ocr", "dev_yarmouk")
TEST_ERROR_SETS = ("T4_tesseract", "T4_abbyy", "T6_tesseract", "T6_openiti", "benchmark")
DESCRIPTIONS = {
    "dev_clean": "clean held-out paragraphs (damage)",
    "dev_typed": "held-out paragraphs with typed-error noise",
    "dev_ocr": "development share of the OCR training pairs",
    "dev_yarmouk": "Yarmouk development split, real scans",
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


def _calibration_counts(job) -> List[List[float]]:
    src, ref, prediction, bins = job
    table = [[0, 0, 0.0] for _ in range(bins)]  # proposals, correct, summed confidence
    gold = derive_labels(src, ref).chars
    for char, label, prob, wanted in zip(src, prediction.chars, prediction.probs, gold):
        if label == KEEP or not is_editable(char):
            continue
        row = table[min(int(prob * bins), bins - 1)]
        row[0] += 1
        row[1] += label == wanted
        row[2] += prob
    return table


def calibration(pool, items: Sequence[Item], predictions: Sequence[Prediction], bins: int = 10):
    """Expected calibration error of the proposed edits and a reliability table.

    Every editable character whose most likely label is an edit counts once:
    correct if that label is the one that turns the source into the reference.
    """
    table = [[0, 0, 0.0] for _ in range(bins)]
    jobs = [(src, ref, p, bins) for (src, ref, _), p in zip(items, predictions)]
    for counts in pool.map(_calibration_counts, jobs, chunksize=8):
        for row, add in zip(table, counts):
            for k in range(3):
                row[k] += add[k]
    total = sum(row[0] for row in table)
    ece = sum(abs(row[1] - row[2]) for row in table) / total if total else 0.0
    rows = [{"from": b / bins, "to": (b + 1) / bins, "edits": row[0],
             "accuracy": row[1] / row[0] if row[0] else None,
             "confidence": row[2] / row[0] if row[0] else None} for b, row in enumerate(table)]
    return ece, total, rows


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


def gates(test: Dict[str, Score], pages: Tuple[int, int], ece: Optional[float]) -> List[Dict]:
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
        gate("Real typed errors (T-1): edit precision", "≥ 0.90", "T-1 not built yet", None),
        gate("Real OCR, modern (T-4): word errors removed", "≥ 20%",
             f"{wer_drop:.1%} (WER {t4_rates['wer_in']:.1%} → {t4_rates['wer_out']:.1%})", wer_drop >= 0.20),
        gate("Real OCR, modern (T-4): edit precision", "≥ 0.85",
             f"{t4.precision:.3f} (harmful edits {t4.harmful:.1%})", t4.precision >= 0.85),
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
             "no edits" if ece is None else f"{ece:.3f}", None if ece is None else ece <= 0.05),
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
                 scores: Dict[str, Dict[float, Score]], gate_rows: List[Dict], ece_rows, ece_edits: int,
                 categories: Score, examples: Dict[str, list], seconds: float) -> None:
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
    parser.add_argument("--model", type=Path, default=Path("artifacts/correct/model"))
    parser.add_argument("--out", type=Path, default=Path("artifacts/eval/correct"))
    parser.add_argument("--text", type=Path, default=Path("data/v1/pretrain"))
    parser.add_argument("--testsets", type=Path, default=Path("data/v1/testsets"))
    parser.add_argument("--ocr_pairs", type=Path, nargs="+",
                        default=[Path("data/v1/pairs/ocr_render.jsonl"), Path("data/v1/pairs/yarmouk_train.jsonl")])
    parser.add_argument("--benchmark", type=Path, default=Path("benchmarks/real_world.csv"))
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
    ece, ece_edits, ece_rows = calibration(pool, error_items, error_predictions)
    pool.close()
    gate_rows = gates(test, pages, ece if ece_edits else None)
    examples = {name: harmful_examples(sets[name], outputs[name], 8) for name in TEST_ERROR_SETS + ("T7_msa",)}

    args.out.mkdir(parents=True, exist_ok=True)
    write_report(args.out / "report.md", args.model, threshold, reason, sets, scores, gate_rows, ece_rows,
                 ece_edits, categories, examples, time.time() - started)
    results = {"model": str(args.model), "threshold": threshold, "reason": reason, "gates": gate_rows,
               "pages_no_worse": pages, "ece": ece, "calibration": ece_rows,
               "scores": {name: {str(t): s.summary() for t, s in by_t.items()} for name, by_t in scores.items()},
               "t4_categories": {k: dict(v) for k, v in categories.by_category.items()}}
    (args.out / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n".join(f"{g['result']:>12s}  {g['gate']}: {g['measured']} (target {g['target']})" for g in gate_rows))
    print(f"report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
