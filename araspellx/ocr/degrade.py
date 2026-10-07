"""Degrade rendered page images like real scans (runs in the OCR container).

Each image gets a random combination of common scan defects at random
strengths: blur, sensor noise, low resolution, JPEG compression, thin or
heavy ink, uneven background, binarization and a slight skew. The goal is
OCR errors that look like the real ones in Yarmouk (checked separately).
"""
from __future__ import annotations

import random
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image


def degrade(image: Image.Image, rng: random.Random) -> Tuple[Image.Image, Dict[str, float]]:
    page = np.asarray(image, dtype=np.float32)
    params: Dict[str, float] = {}

    if rng.random() < 0.5:  # ink thickness
        size = rng.choice([2, 3])
        kernel = np.ones((size, size), np.uint8)
        page = (cv2.erode if rng.random() < 0.6 else cv2.dilate)(page, kernel)
        params["ink"] = size
    if rng.random() < 0.5:
        params["skew"] = rng.uniform(-1.2, 1.2)
        h, w = page.shape
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), params["skew"], 1.0)
        page = cv2.warpAffine(page, matrix, (w, h), borderValue=255)
    if rng.random() < 0.6:
        params["scale"] = rng.uniform(0.35, 0.8)  # scanned at low resolution
        h, w = page.shape
        small = cv2.resize(page, (int(w * params["scale"]), int(h * params["scale"])),
                           interpolation=cv2.INTER_AREA)
        page = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5:
        params["blur"] = rng.uniform(0.5, 1.8)
        page = cv2.GaussianBlur(page, (0, 0), params["blur"])
    if rng.random() < 0.4:  # uneven paper background
        params["background"] = rng.uniform(10, 50)
        h, w = page.shape
        gradient = np.linspace(0, params["background"], w, dtype=np.float32)[None, :].repeat(h, 0)
        page = np.minimum(page, 255 - gradient)
    if rng.random() < 0.6:
        params["noise"] = rng.uniform(4, 25)
        page = page + np.random.default_rng(rng.randrange(1 << 30)).normal(0, params["noise"], page.shape)
    page = np.clip(page, 0, 255).astype(np.uint8)
    if rng.random() < 0.3 and params.get("noise", 0) < 12 and params.get("background", 0) < 25:
        params["binarize"] = 1  # on noisy pages binarization destroys the text entirely
        page = cv2.adaptiveThreshold(page, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15)
    if rng.random() < 0.5:
        params["jpeg"] = rng.randint(15, 60)
        ok, encoded = cv2.imencode(".jpg", page, [cv2.IMWRITE_JPEG_QUALITY, params["jpeg"]])
        page = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return Image.fromarray(page), params
