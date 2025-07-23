import os
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=os.getenv("HF_API_TOKEN")
)

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

def is_response_broken(text):
    weirdness_score = sum(1 for c in text if ord(c) > 1000 or c in "¼½¾™®©•§µ")
    has_fake_q = any(kw in text.lower() for kw in ["user:", "question:", "q:", "can you explain more about"])
    return (
        len(text) > 2000 or
        weirdness_score > 5 or
        has_fake_q or
        any(bad in text.lower() for bad in [
            "retrieved from", "last revised", "wikipedia.org",
            "porn", "xxx", "tube8", "custom essay"
        ])
    )

def generate_zephyr_answer(context, question, history=None):
    history_prompt = ""
    if history:
        last_n = history[-1:] if len(history) >= 1 else history
        for turn in last_n:
            history_prompt += f"{turn['user']}\n{turn['bot']}\n"

    wants_detail = any(keyword in question.lower() for keyword in DETAIL_KEYWORDS)
    last_user_turn = history[-1]["user"] if history else ""

    if question.strip().lower() in ["give an example", "can you give an example?"] and last_user_turn:
        effective_question = f"{last_user_turn} {question}"
    elif wants_detail and len(question.split()) <= 8 and last_user_turn:
        effective_question = f"{last_user_turn} {question}"
    else:
        effective_question = question

    style_instruction = (
        "If the user asks for more detail or an example, respond with 3–5 concise sentences. "
        "Include one focused example relevant to the recent question. Do not go off-topic or provide multiple examples."
        if wants_detail
        else "Answer clearly and concisely in 2–3 sentences. Do not add extra explanation unless asked."
    )

    prompt = f"""
You are a helpful and knowledgeable AI assistant.

{style_instruction}

Use the context and short chat history below to answer the user's current question only.

Do not invent new questions.
Do not continue the conversation unless asked.
Avoid repeating or generating follow-up questions.

Context:
{context}

Recent exchange:
{history_prompt}

User question:
{effective_question}
""".strip()

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-alpha",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and concise assistant. Do not ask your own questions. Do not continue the conversation. Only respond directly to the user input."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=300
        )

        if not response or not response.choices or not response.choices[0].message:
            return "⚠️ The assistant could not generate a valid response. Please try again."

        answer = response.choices[0].message.content.strip()

        # Temizlik: “Assistant:” başlığını sil
        if answer.lower().startswith("assistant:"):
            answer = answer[len("assistant:"):].strip()

        # Assistant kendi kendine soru üretmişse temizle
        if is_response_broken(answer):
            return "⚠️ The assistant generated an invalid or off-topic response. Please rephrase your question."

        return answer

    except Exception as e:
        return f"⚠️ Error during API call: {e}"
