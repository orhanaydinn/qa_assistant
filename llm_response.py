import os
import requests

def generate_zephyr_answer(context, question, history=None):
    """
    Hugging Face Zephyr 7B modeline API ile bağlanarak
    context ve history ile birlikte cevap üretir.
    """
    api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-alpha"
    headers = {
        "Authorization": f"Bearer {os.getenv('HF_API_TOKEN')}"
    }

    # History varsa prompt içine dahil et
    history_prompt = ""
    if history:
        for pair in history:
            history_prompt += f"User: {pair['user']}\nAssistant: {pair['bot']}\n"

    full_prompt = f"""{history_prompt}
Answer the following question based on the context.

Context:
{context}

Question: {question}
Answer:"""

    payload = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 250,
            "temperature": 0.7
        }
    }

    response = requests.post(api_url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            output = response.json()[0]["generated_text"]
            return output.split("Answer:")[-1].strip()
        except Exception:
            return "⚠️ Unable to parse the model output."
    else:
        return f"⚠️ API Error {response.status_code}: {response.text}"
