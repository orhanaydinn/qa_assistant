# -*- coding: utf-8 -*-
"""
OCR utilities using EasyOCR (English only)
"""

import numpy as np
from PIL import Image
import easyocr

# EasyOCR reader — sadece İngilizce
reader = easyocr.Reader(['en'], gpu=False)


def extract_ocr_chunks(image):
    """
    Perform OCR (English only) using EasyOCR on a PIL Image or file-like object.

    Args:
        image: PIL.Image.Image, file path, or file-like object

    Returns:
        List[str]: OCR'dan elde edilen metin parçaları
    """
    # Eğer PIL değilse, PIL'e çevir
    if not isinstance(image, Image.Image):
        image = Image.open(image).convert("RGB")
    else:
        image = image.convert("RGB")

    # EasyOCR numpy array ister
    img_np = np.array(image)

    # OCR yap
    results = reader.readtext(img_np)

    # Sadece text alanlarını döndür
    chunks = []
    for _, text, conf in results:
        text = text.strip()
        if text:
            chunks.append(text)

    return chunks
