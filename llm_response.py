import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=os.getenv("HF_API_TOKEN")
)

# Genişletilmiş detay anahtar kelime listesi
DETAIL_KEYWORDS = [
    "example", "give an example", "show an example", "for instance",
    "explain more", "explain further", "more detail", "detailed",
    "can you elaborate", "elaborate", "expand", "walk me through",
    "how does it work", "how does this work", "what does that mean",
    "what do you mean", "clarify", "can you clarify", "describe",
    "in depth", "step by step", "demonstrate", "could you show",
    "show me how", "could you explain that better", "break it down",
    "break down", "go deeper", "give more information", "deep dive",
    "drill down", "case study", "use case", "sample use", "walkthrough",
    "further explanation", "what's an example", "what's a use case", "specific scenario"
]

# Cevap güvenlik ve tutarlılık kontrolü
def is_response_broken(text):
    return (
        len(text) > 1500 or
        any(bad in text.lower() for bad in [
            "retrieved from", "last revised", "wikipedia.org", "porn", "xxx", "tube8", "custom essay"
        ]) or
        text.count(" ") < 10
    )

def generate_zephyr_answer(context, question, history=None):
    history_prompt = ""
    if history:
        for turn in history:
            history_prompt += f"User: {turn['user']}\nAssistant: {turn['bot']}\n"

    # Detay istiyor mu?
    wants_detail = any(keyword in question.lower() for keyword in DETAIL_KEYWORDS)

    if wants_detail:
        style_instruction = "Provide a more detailed and expanded answer. Include an example if applicable."
    else:
        style_instruction = "Answer clearly and concisely in 2–3 sentences. Do not add extra explanation unless asked."

    prompt = f"""
You are a helpful and knowledgeable AI assistant.

{style_instruction}

Use the context below to answer the user's question.

Do not invent questions or go off-topic.
Avoid inappropriate or unrelated content.
Do not exceed 3–5 sentences in total.

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
            max_tokens=250  # sınırlandırılmış üretim
        )
        answer = response.choices[0].message.content.strip()

        if is_response_broken(answer):
            return "The assistant encountered an error generating a reliable response. Please try rephrasing your question."

        return answer

    except Exception as e:
        return f"Error during API call: {e}"
