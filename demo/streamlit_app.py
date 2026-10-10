"""The AraSpellX demo: a web page to try the model on your own text, as a Streamlit app.

    pip install -r demo/requirements.txt && streamlit run demo/streamlit_app.py

Streamlit Community Cloud installs demo/requirements.txt and runs this file from
the root of the repository; the model is downloaded from the Hugging Face Hub on
the first visit. The page is plain HTML and CSS in a Streamlit component, which
sends the form to Python and shows the corrections.
"""
from __future__ import annotations

import copy
import html
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # the araspellx package next to this folder

from araspellx.correct.corrector import HUB_MODEL, Corrector, Result  # noqa: E402

MAX_CHARS = 5000  # longest text the page accepts (about 900 words), so one visitor cannot hold the server

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

# Streamlit's header, toolbar and padding hidden, the page's background behind it
APP_STYLE = """<style>
 header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"], footer { display: none !important; }
 .stApp { background: #fafafa; }
 [data-testid="stMainBlockContainer"] { padding: 0; max-width: none; }
</style>"""
# The page sits in a shadow root: it starts from the browser's defaults, as on a page of its own
PAGE_STYLE = """
 :host { all: initial; display: block; }
 .page { font-family: "Segoe UI", Tahoma, sans-serif; max-width: 920px; margin: 2em auto; padding: 0 1em; color: #222; }
 textarea, input, button { font-family: inherit; }  /* browsers give form fields a monospace font of their own */
 h1 { margin-bottom: 0.2em; } .sub { color: #666; margin-top: 0; }
 textarea { width: 100%; height: 9em; font-size: 1.25em; padding: 0.6em; box-sizing: border-box; }
 .row { display: flex; gap: 1.5em; align-items: center; flex-wrap: wrap; margin: 0.8em 0; }
 button { font-size: 1.1em; padding: 0.4em 1.6em; cursor: pointer; }
 .box { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 1em 1.2em; font-size: 1.3em; line-height: 2.1; }
 .old { background: #fde2e2; border-bottom: 2px solid #d33; } .new { background: #dff5e1; border-bottom: 2px solid #2a2; }
 table { border-collapse: collapse; width: 100%; background: #fff; font-size: 1.1em; }
 td, th { border: 1px solid #ddd; padding: 0.4em 0.7em; text-align: right; } th { background: #f0f0f0; }
 .ex { font-size: 0.9em; padding: 0.2em 0.8em; } .meta { color: #888; font-size: 0.85em; margin-top: 2em; }
"""
PAGE = """<div class="page" lang="ar" dir="rtl">
<h1>AraSpellX</h1><p class="sub">تصحيح الأخطاء الإملائية وأخطاء المسح الضوئي في النصوص العربية</p>
<form method="post">
<textarea name="text" maxlength="{max_chars}" placeholder="اكتب أو الصق نصا عربيا هنا…">{text}</textarea>
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
</div>"""
# Shows the page, makes the examples fill the text area and sends the form to Python
PAGE_SCRIPT = """
export default function ({ data, parentElement, setTriggerValue }) {
    let root = parentElement.querySelector(".araspellx");
    if (!root) {
        root = document.createElement("div");
        root.className = "araspellx";
        parentElement.appendChild(root);
    }
    if (root.shown === data) return;  // the same page again: keep what the visitor typed
    root.shown = root.innerHTML = data;
    const form = root.querySelector("form");
    for (const button of root.querySelectorAll(".ex"))
        button.onclick = () => form.elements.text.value = button.dataset.text;
    form.onsubmit = (event) => {
        event.preventDefault();
        const { text, source, threshold } = form.elements;
        setTriggerValue("submit", { text: text.value, source: source.value, threshold: threshold.value });
    };
}
"""


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


def render_page(corrector: Corrector, text: str = "", source: str = "typed",
                threshold: Optional[float] = None, result: str = "") -> str:
    """The page with the form filled in and the result HTML below it."""
    examples = " ".join(f'<button type="button" class="ex" data-text="{html.escape(t)}">{name}</button>'
                        for name, t in EXAMPLES)
    meta = (f"النموذج: {HUB_MODEL} · الجهاز: {corrector.device} · "
            "يُعالَج النص على الجهاز الذي يشغّل هذه الصفحة ولا يُحفظ")
    return PAGE.format(text=html.escape(text), typed="checked" if source == "typed" else "",
                       ocr="checked" if source == "ocr" else "", max_chars=MAX_CHARS,
                       threshold=corrector.threshold if threshold is None else threshold,
                       examples=examples, result=result, meta=meta)


def correct_form(corrector: Corrector, text: str, source: str, threshold) -> Tuple[str, str, float, str]:
    """Check the submitted form values and correct the text: (text, source, threshold, result HTML)."""
    text = text[:MAX_CHARS]
    source = source if source in ("typed", "ocr") else "typed"
    try:
        threshold = min(max(float(threshold), 0.5), 0.99)
    except (TypeError, ValueError):
        threshold = corrector.threshold
    visitor = copy.copy(corrector)  # shares the model; the threshold is this visitor's alone
    visitor.threshold = threshold
    started = time.time()
    result = visitor.correct(text, source)
    return text, source, threshold, render_result(text, result, time.time() - started)


@st.cache_resource(show_spinner="Loading the model…")
def load_corrector() -> Corrector:
    return Corrector(HUB_MODEL, device="cpu")


@st.cache_resource
def page_component():
    return st.components.v2.component("araspellx_page", css=PAGE_STYLE, js=PAGE_SCRIPT)


def on_submit() -> None:
    form = st.session_state.page.submit
    st.session_state.form = correct_form(corrector, form["text"], form["source"], form["threshold"])


st.set_page_config(page_title="AraSpellX", layout="wide")
st.html(APP_STYLE)
corrector = load_corrector()
page_component()(key="page", data=render_page(corrector, *st.session_state.get("form", ())),
                 on_submit_change=on_submit)
