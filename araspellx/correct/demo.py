"""Try a trained model on your own text, in the browser or the terminal.

    python -m araspellx.correct.demo                         # web page at http://localhost:8000
    python -m araspellx.correct.demo --text "ذهبت الي الجامعه"  # one text in the terminal
    python -m araspellx.correct.demo --cli                   # type texts in the terminal

The web page shows Arabic right to left; terminals often do not. Everything
runs locally: the text never leaves the computer.
"""
from __future__ import annotations

import argparse
import html
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from araspellx.correct.corrector import Corrector, Result

CATEGORIES = {
    "hamza_alef": "همزة على الألف", "hamza_seat": "كرسي الهمزة", "ta_marbuta": "التاء المربوطة",
    "alef_maqsura": "الألف المقصورة", "alef_fariqa": "الألف الفارقة", "dots": "النقاط",
    "spacing": "المسافات", "typo": "خطأ كتابي", "mixed": "أكثر من خطأ",
}
EXAMPLES = [
    ("أخطاء كتابة", "ذهبت الي الجامعه صباحا لكي احضر المحاضره الاولي، ثم قابلت صديقي في المكتبه."),
    ("أخطاء شائعة", "اعتقد ان هذه الفكره جيده جدا ولاكن يجب ان نفكر فيها مره اخري قبل التنفيذ."),
    ("أخطاء مسح ضوئي", "تعتبر المدينه من أهم المراكز النجارية في المنطفة، حيث يقصدها الزوار من حميع أنحاء العالم."),
    ("نص سليم", "يعد التعليم من أهم ركائز التنمية في المجتمعات الحديثة."),
]

PAGE = """<!doctype html>
<html lang="ar" dir="rtl"><head><meta charset="utf-8"><title>AraSpellX</title>
<style>
 body {{ font-family: "Segoe UI", Tahoma, sans-serif; max-width: 920px; margin: 2em auto; padding: 0 1em; background: #fafafa; color: #222; }}
 h1 {{ margin-bottom: 0.2em; }} .sub {{ color: #666; margin-top: 0; }}
 textarea {{ width: 100%; height: 9em; font-size: 1.25em; padding: 0.6em; box-sizing: border-box; }}
 .row {{ display: flex; gap: 1.5em; align-items: center; flex-wrap: wrap; margin: 0.8em 0; }}
 button {{ font-size: 1.1em; padding: 0.4em 1.6em; cursor: pointer; }}
 .box {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1em 1.2em; font-size: 1.3em; line-height: 2.1; }}
 .old {{ background: #fde2e2; border-bottom: 2px solid #d33; }} .new {{ background: #dff5e1; border-bottom: 2px solid #2a2; }}
 table {{ border-collapse: collapse; width: 100%; background: #fff; font-size: 1.1em; }}
 td, th {{ border: 1px solid #ddd; padding: 0.4em 0.7em; text-align: right; }} th {{ background: #f0f0f0; }}
 .ex {{ font-size: 0.9em; padding: 0.2em 0.8em; }} .meta {{ color: #888; font-size: 0.85em; margin-top: 2em; }}
</style></head><body>
<h1>AraSpellX</h1><p class="sub">تصحيح الأخطاء الإملائية وأخطاء المسح الضوئي في النصوص العربية</p>
<form method="post">
<textarea name="text" placeholder="اكتب أو الصق نصا عربيا هنا…">{text}</textarea>
<div class="row">
 <label><input type="radio" name="source" value="typed" {typed}> نص مكتوب</label>
 <label><input type="radio" name="source" value="ocr" {ocr}> نص من مسح ضوئي (OCR)</label>
 <label>الحد الأدنى للثقة <input type="number" name="threshold" min="0.5" max="0.99" step="0.01" value="{threshold}" style="width:5em"></label>
 <button type="submit">صحّح</button>
</div>
<div class="row">أمثلة: {examples}</div>
</form>
{result}
<p class="meta">{meta}</p>
<script>function ex(t) {{ document.querySelector('textarea').value = t; }}</script>
</body></html>"""


def render_result(text: str, result: Result, seconds: float) -> str:
    if not result.corrections:
        return f'<h2>النتيجة</h2><div class="box">{html.escape(result.text)}</div><p>لم يجد النموذج ما يصححه بثقة كافية ({seconds * 1000:.0f} ms).</p>'
    before, position = [], 0
    for c in result.corrections:
        before.append(html.escape(text[position:c.start]))
        before.append(f'<span class="old">{html.escape(text[c.start:c.end])}</span>')
        position = c.end
    before.append(html.escape(text[position:]))
    after = []
    for segment, c in result.segments:
        if c is None:
            after.append(html.escape(segment))
        else:
            word = segment.rstrip()
            after.append(f'<span class="new" title="{html.escape(c.original)} ← {c.confidence:.0%}">'
                         f'{html.escape(word)}</span>{html.escape(segment[len(word):])}')
    rows = "".join(f"<tr><td>{i}</td><td>{html.escape(c.original)}</td><td>{html.escape(c.replacement)}</td>"
                   f"<td>{c.confidence:.0%}</td><td>{CATEGORIES.get(c.category, c.category)}</td></tr>"
                   for i, c in enumerate(result.corrections, 1))
    return (f'<h2>قبل</h2><div class="box">{"".join(before)}</div>'
            f'<h2>بعد</h2><div class="box">{"".join(after)}</div>'
            f'<h2>التصحيحات ({len(result.corrections)}، {seconds * 1000:.0f} ms)</h2>'
            f"<table><tr><th>#</th><th>قبل</th><th>بعد</th><th>الثقة</th><th>النوع</th></tr>{rows}</table>")


def serve(corrector: Corrector, port: int, model: Path, open_browser: bool) -> None:
    def page(text: str = "", source: str = "typed", threshold: float = corrector.threshold, result: str = "") -> bytes:
        examples = " ".join(f'<button type="button" class="ex" onclick="ex({html.escape(repr(t))})">{name}</button>'
                            for name, t in EXAMPLES)
        meta = (f"النموذج: {html.escape(str(model))} · الجهاز: {corrector.device} · "
                f"يعمل محليا، لا يغادر النص هذا الجهاز")
        return PAGE.format(text=html.escape(text), typed="checked" if source == "typed" else "",
                           ocr="checked" if source == "ocr" else "", threshold=threshold,
                           examples=examples, result=result, meta=meta).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes) -> None:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(page())

        def do_POST(self):
            form = urllib.parse.parse_qs(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8"))
            text = form.get("text", [""])[0]
            source = form.get("source", ["typed"])[0]
            threshold = float(form.get("threshold", [corrector.threshold])[0])
            default, corrector.threshold = corrector.threshold, threshold
            started = time.time()
            result = corrector.correct(text, source)
            corrector.threshold = default
            self._send(page(text, source, threshold, render_result(text, result, time.time() - started)))

        def log_message(self, *args):  # keep the terminal quiet
            pass

    url = f"http://localhost:{port}"
    print(f"AraSpellX demo running at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        webbrowser.open(url)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()


def show(corrector: Corrector, text: str, source: str) -> None:
    result = corrector.correct(text, source)
    print(result.text)
    for c in result.corrections:
        print(f"  {c.original} -> {c.replacement}   ({c.confidence:.0%}, {c.category}, characters {c.start}-{c.end})")
    if not result.corrections:
        print("  (no corrections)")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", type=Path, default=Path("artifacts/correct/best_model"))
    parser.add_argument("--text", help="correct this text and exit")
    parser.add_argument("--cli", action="store_true", help="type texts in the terminal")
    parser.add_argument("--source", choices=["typed", "ocr"], default="typed")
    parser.add_argument("--threshold", type=float, default=None, help="default: the one chosen by the evaluation")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no_browser", action="store_true")
    parser.add_argument("--device", default=None, help="cpu or cuda (default: cuda if available)")
    args = parser.parse_args()

    corrector = Corrector(args.model, device=args.device, threshold=args.threshold)
    if args.text:
        show(corrector, args.text, args.source)
    elif args.cli:
        print("Type a text and press Enter (an empty line exits).")
        while text := input("> ").strip():
            show(corrector, text, args.source)
    else:
        serve(corrector, args.port, args.model, not args.no_browser)


if __name__ == "__main__":
    main()
