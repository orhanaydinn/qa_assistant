# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:17:29 2025

@author: Orhan
"""

# llm_response.py
import requests
import os

API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
headers = {"Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"}

def generate_answer(context, question):
    prompt = f"Use the following context to answer the question:\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
    payload = {
        "inputs": prompt,
        "parameters": {
            "temperature": 0.7,
            "max_new_tokens": 256
        }
    }

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code == 200:
        output = response.json()
        return output[0]["generated_text"].split("Answer:")[-1].strip()
    else:
        return f"Error: {response.status_code} - {response.text}"
