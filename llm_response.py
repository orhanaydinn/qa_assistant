import os
import requests

def generate_zephyr_answer(context, question):
    api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"
    }

    prompt = f"""Answer the question based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"""

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 200,
            "temperature": 0.7
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()[0]["generated_text"].split("Answer:")[-1].strip()
    else:
        return f"⚠️ Error: {response.status_code} - {response.text}"
