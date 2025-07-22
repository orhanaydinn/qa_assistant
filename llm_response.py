import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=os.getenv("HF_API_TOKEN")
)

def generate_zephyr_answer(context, question, history=None):
    # History prompt'u hazırla
    history_prompt = ""
    if history:
        for turn in history:
            history_prompt += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

    prompt = f"""
You are a helpful assistant that always provides clear, practical, and detailed answers based on the document provided.

If the user asks for an example, try to give a relevant and realistic one that relates to the topic.

Here is the previous conversation:
{history_prompt}

Here is some context from the document:
{context}

Question:
{question}

Answer:"""


    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-alpha",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error during API call: {e}"
