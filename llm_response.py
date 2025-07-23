import os
import requests
from huggingface_hub import InferenceClient

# Hugging Face Zephyr istemcisi
client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=os.getenv("HF_API_TOKEN")
)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")


# --- 1. Web özeti alma (Serper.dev) ---
def get_web_summary_serper(query):
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        data = {"q": query}
        res = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=8)
        if res.status_code != 200:
            return ""
        json_data = res.json()

        snippets = []
        if "answerBox" in json_data and "answer" in json_data["answerBox"]:
            snippets.append(json_data["answerBox"]["answer"])
        for item in json_data.get("organic", [])[:2]:
            if "snippet" in item:
                snippets.append(item["snippet"])
        return " ".join(snippets).strip()
    except Exception:
        return ""


# --- 2. Web gerektiren kelimeleri tespit et ---
def needs_web_context(question):
    keywords = [
        "2024", "2023", "today", "now", "current", "latest", "this year", "recent",
        "weather", "election", "result", "news", "price", "score", "exchange rate",
        "who won", "when is", "what time", "live", "president", "prime minister"
    ]
    return any(kw in question.lower() for kw in keywords)


# --- 3. Cevap bozuk mu? (çok katı değil) ---
def is_response_broken(text):
    too_long = len(text) > 2000
    weird_chars = sum(1 for c in text if ord(c) > 1000) > 5
    banned_fragments = ["porn", "xxx", "custom essay", "retrieved from", "last revised", "wikipedia.org"]
    return too_long or weird_chars or any(b in text.lower() for b in banned_fragments)


# --- 4. Fazla yapıyı temizle ---
def clean_bad_patterns(text):
    bad_tokens = ["user:", "question:", "q:", "note:", "recent exchange:"]
    for token in bad_tokens:
        if token in text.lower():
            return text.split(token)[0].strip()
    return text


# --- 5. Cevap üretici ana fonksiyon ---
def generate_zephyr_answer(context, question, history=None):
    status_message = "Generating answer..."

    if needs_web_context(question):
        web_info = get_web_summary_serper(question)
        if web_info.strip():
            context = f"[WEB RESULT]\n{web_info.strip()}\n\n{context}"
            status_message = "Searching the internet..."
        else:
            return "⚠️ No reliable up-to-date information was found online.", "Searching the internet..."

    # Son geçmiş (son 1 mesaj yeterli)
    history_prompt = ""
    if history:
        for turn in history[-1:]:
            history_prompt += f"{turn['user']}\n{turn['bot']}\n"

    # Kullanıcı detay istiyor mu?
    DETAIL_KEYWORDS = [
        "example", "give an example", "show an example", "explain more", "more detail",
        "elaborate", "expand", "walk me through", "how does", "clarify", "describe", "in depth"
    ]
    wants_detail = any(k in question.lower() for k in DETAIL_KEYWORDS)
    last_q = history[-1]["user"] if history else ""

    if wants_detail and len(question.split()) <= 8 and last_q:
        effective_question = f"{last_q} {question}"
    else:
        effective_question = question

    style_instruction = (
        "If the user asks for more detail or an example, respond with 3–5 concise sentences. One example only."
        if wants_detail else
        "Answer in 2–3 concise sentences. Do not elaborate unless requested."
    )

    # Prompt oluştur
    prompt = f"""
You are a helpful and factual assistant.
{style_instruction}

Context:
{context}

Recent exchange:
{history_prompt}

User question:
{effective_question}
""".strip()

    # API çağrısı
    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-alpha",
            messages=[
                {"role": "system", "content": "You are a concise and accurate assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        if answer.lower().startswith("assistant:"):
            answer = answer[len("assistant:"):].strip()

        answer = clean_bad_patterns(answer)

        if is_response_broken(answer):
            return "⚠️ The assistant generated an invalid response. Please try rephrasing.", status_message

        return answer, status_message

    except Exception as e:
        return f"⚠️ Error during API call: {e}", status_message
