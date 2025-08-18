# image_gen.py
# -*- coding: utf-8 -*-
"""
Image generation via Hugging Face InferenceClient (Streamlit Cloud ready).
- Token: st.secrets["HF_API_TOKEN"]
- Default model: stabilityai/stable-diffusion-xl-base-1.0
- Returns: PNG bytes (for inline <img> in app.py)
"""

from __future__ import annotations
import io
import random
import time
from typing import Optional

import streamlit as st
from huggingface_hub import InferenceClient
from PIL import Image

# ---- Secrets / Config ----
HF_API_TOKEN = st.secrets.get("HF_API_TOKEN", "")
if not HF_API_TOKEN:
    # İstersen alternatif isim de destekle
    HF_API_TOKEN = st.secrets.get("HF_TOKEN", "")

PRIMARY_MODEL = st.secrets.get("IMAGE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")

# İsteğe bağlı fallback modeller
FALLBACK_MODELS = [
    PRIMARY_MODEL,
    "stabilityai/stable-diffusion-2-1",
    "runwayml/stable-diffusion-v1-5",
]

class ImageGenError(RuntimeError):
    pass


def _client(model_name: str) -> InferenceClient:
    if not HF_API_TOKEN:
        raise ImageGenError("HF_API_TOKEN not set in Streamlit secrets.")
    return InferenceClient(model=model_name, token=HF_API_TOKEN)


def _pil_to_png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def generate_image_from_prompt(
    prompt: str,
    negative_prompt: Optional[str] = None,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
    width: int = 1024,
    height: int = 1024,
    seed: Optional[int] = None,
    max_retries: int = 3,
    warmup_delay: float = 2.0,
) -> bytes:
    """
    Text-to-image üretir (PNG bytes döner).
    SDXL destekli; gerekirse listedeki fallback modellere dener.
    """
    if not prompt or not prompt.strip():
        raise ImageGenError("Prompt is empty.")

    last_err = ""
    rng_seed = seed if seed is not None else random.randint(0, 2**31 - 1)

    for attempt in range(1, max_retries + 1):
        for model_name in FALLBACK_MODELS:
            try:
                client = _client(model_name)
                # InferenceClient.text_to_image -> PIL.Image döner
                img = client.text_to_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=rng_seed,
                )
                # PNG bytes olarak dön
                return _pil_to_png_bytes(img)

            except Exception as e:
                last_err = f"{model_name}: {e}"
                # İlk istekte “Model is loading” olabilir; kısa bekleme faydalı
                time.sleep(warmup_delay)
                continue

        # Tüm modeller denendi, backoff ile tekrar
        warmup_delay *= 1.75

    raise ImageGenError(f"Image generation failed after {max_retries} attempts. Last error: {last_err}")


# CLI/REPL hızlı test
if __name__ == "__main__":
    try:
        png = generate_image_from_prompt(
            "a red balloon on a white background, minimal, no text",
            guidance_scale=7.0,
            num_inference_steps=25,
            width=768,
            height=768,
        )
        with open("test_output.png", "wb") as f:
            f.write(png)
        print("Saved test_output.png")
    except Exception as e:
        print("Error:", e)
