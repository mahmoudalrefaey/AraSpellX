# The fixed OCR environment of AraSpellX v1: rendering Arabic text, degrading
# page images and running Tesseract. Everything that produces OCR errors for
# training or testing runs here, so the OCR engine and fonts never change.
#
#   docker build -t araspellx-ocr docker/ocr
#   docker run --rm -v "$PWD:/work" -w /work araspellx-ocr python3 -m araspellx.ocr.check
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONDONTWRITEBYTECODE=1 PYTHONIOENCODING=utf-8

RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
        libraqm0 libfribidi0 libharfbuzz0b \
        fonts-hosny-amiri fonts-noto-core fonts-sil-scheherazade \
        python3 python3-pip ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Tesseract's most accurate Arabic model, pinned to the tessdata_best 4.1.0 release.
RUN curl -fsSL -o /usr/share/tesseract-ocr/5/tessdata/ara.traineddata \
        https://github.com/tesseract-ocr/tessdata_best/raw/4.1.0/ara.traineddata

RUN pip3 install --no-cache-dir --break-system-packages \
        pillow==11.3.0 numpy==1.26.4 opencv-python-headless==4.10.0.84 rapidfuzz==3.14.5
