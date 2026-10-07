"""Render Arabic paragraphs as page images whose exact text is known.

Runs in the OCR container (Pillow with libraqm for Arabic shaping).
Paragraphs are word-wrapped right to left; line breaks correspond to
spaces of the paragraph, so OCR output of the image, with its line breaks
turned into spaces, can be compared directly with the paragraph.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from PIL import Image, ImageDraw, ImageFont

# Open (OFL) Arabic fonts installed in the OCR container.
FONTS = {
    "amiri": "/usr/share/fonts/opentype/fonts-hosny-amiri/Amiri-Regular.ttf",
    "noto-naskh": "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "noto-sans": "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
    "noto-kufi": "/usr/share/fonts/truetype/noto/NotoKufiArabic-Regular.ttf",
    "scheherazade": "/usr/share/fonts/truetype/scheherazade/Scheherazade-Regular.ttf",
}


@dataclass
class Page:
    image: Image.Image
    lines: List[str]


def wrap(text: str, draw: ImageDraw.ImageDraw, font: ImageFont.FreeTypeFont, width: int) -> List[str]:
    lines, current = [], []
    for word in text.split(" "):
        candidate = " ".join(current + [word])
        if current and draw.textlength(candidate, font=font, direction="rtl", language="ar") > width:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))
    return lines


def render(text: str, font: str = "amiri", points: float = 14, dpi: int = 300,
           width_inches: float = 6.0, margin: int = 60, line_spacing: float = 1.6) -> Page:
    """Render a paragraph right-aligned, black on white, at the given resolution."""
    size = round(points * dpi / 72)
    face = ImageFont.truetype(FONTS[font], size, layout_engine=ImageFont.Layout.RAQM)
    width = round(width_inches * dpi)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines = wrap(text, probe, face, width - 2 * margin)
    step = round(size * line_spacing)
    image = Image.new("L", (width, 2 * margin + step * len(lines)), 255)
    draw = ImageDraw.Draw(image)
    for i, line in enumerate(lines):
        line_width = draw.textlength(line, font=face, direction="rtl", language="ar")
        draw.text((width - margin - line_width, margin + i * step), line, font=face,
                  fill=0, direction="rtl", language="ar")
    return Page(image, lines)
