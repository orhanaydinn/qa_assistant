from huggingface_hub import InferenceClient
from PIL import Image
import streamlit as st

# 🔧 Text-to-Image (Stable Diffusion XL)
def generate_image_from_prompt(prompt: str) -> Image.Image:
    client = InferenceClient(
        model="stabilityai/stable-diffusion-xl-base-1.0",
        token=st.secrets["HF_API_TOKEN"]
    )

    return client.text_to_image(
        prompt=prompt,
        height=1024,
        width=1024,
        guidance_scale=7.5,
        num_inference_steps=30
    )
