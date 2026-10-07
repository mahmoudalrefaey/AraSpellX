"""Smoke test of the OCR container: render a paragraph in every font and OCR it.

    docker run --rm -v "$PWD:/work" -w /work araspellx-ocr python3 -m araspellx.ocr.check
"""
from __future__ import annotations

import time

from rapidfuzz.distance import Levenshtein

from araspellx.ocr.render import FONTS, render
from araspellx.ocr.tesseract import ocr

PARAGRAPH = (
    "تعمل الشركة على تطوير نظام جديد لمعالجة البيانات وتحسين جودة الخدمة المقدمة "
    "للعملاء في جميع الفروع، وقد بلغت نسبة النمو 12% خلال عام 2025 بحسب التقرير السنوي "
    "الذي نشرته الإدارة في شهر مارس."
)


def main() -> None:
    for font in FONTS:
        started = time.perf_counter()
        page = render(PARAGRAPH, font=font)
        text = ocr(page.image)
        cer = Levenshtein.distance(text, PARAGRAPH) / len(PARAGRAPH)
        print(f"{font:13s} lines={len(page.lines)} CER={cer:.3f} "
              f"time={time.perf_counter() - started:.1f}s  {text[:90]}")


if __name__ == "__main__":
    main()
