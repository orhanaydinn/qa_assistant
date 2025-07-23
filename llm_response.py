import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=os.getenv("HF_API_TOKEN")
)

DETAIL_KEYWORDS = [
    "example",
    "give an example",
    "show an example",
    "for instance",
    "explain more",
    "explain further",
    "more detail",
    "detailed",
    "can you elaborate",
    "elaborate",
    "expand",
    "walk me through",
    "how does it work",
    "how does this work",
    "what does that mean",
    "what do you mean",
    "clarify",
    "can you clarify",
    "describe",
    "in depth",
    "step by step",
    "demonstrate",
    "could you show",
    "show me how",
    "could you explain that better",
    "break it down",
    "break down",
    "go deeper",
    "give more information",
    "deep dive",
    "drill down",
    "case study",
    "use case",
    "sample use",
    "walkthrough",
    "further explanation",
    "what's an example",
    "what's a use case",
    "specific scenario"
]

def generate_zephyr_answer(context, question, history=None):
    history_prompt = ""
    if history:
        for turn in history:
            history_prompt += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

    # Detay istenip istenmediğini kontrol et
    wants_detail = any(keyword in question.lower() for keyword in DETAIL_KEYWORDS)

    if wants_detail:
        style_instruction = "Provide a more detailed and expanded answer. Include an example if applicable."
    else:
        style_instruction = "Answer clearly and concisely in 2–3 sentences. Do not add extra explanation unless asked."

    prompt = f"""
You are a helpful and knowledgeable AI assistant.

{style_instruction}

Use the context below to answer the user's question.

[CONTEXT]
{context}

[CONVERSATION HISTORY]
{history_prompt}

[QUESTION]
{question}

[ANSWER]
"""

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-alpha",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error during API call: {e}"
