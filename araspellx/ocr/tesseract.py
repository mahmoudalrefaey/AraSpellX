"""Run the project's fixed OCR engine (Tesseract 5, tessdata_best Arabic) on an image."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Union

from PIL import Image

SPACES = re.compile(r"\s+")


def ocr(image: Union[Image.Image, Path], psm: int = 6) -> str:
    """OCR an image; line breaks are joined with spaces (one paragraph of text)."""
    with tempfile.TemporaryDirectory() as tmp:
        if isinstance(image, Image.Image):
            path = Path(tmp) / "page.png"
            image.save(path, dpi=(300, 300))
        else:
            path = Path(image)
        result = subprocess.run(
            ["tesseract", str(path), "stdout", "-l", "ara", "--psm", str(psm)],
            capture_output=True, check=True)
    text = result.stdout.decode("utf-8")
    return SPACES.sub(" ", text).strip()
