import os
import requests
import streamlit as st
from huggingface_hub import InferenceClient

HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
SERPER_API_KEY = st.secrets["SERPER_API_KEY"]
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-alpha", token=HF_API_TOKEN)

def get_web_summary_serper(query):
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        data = {"q": query}
        res = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=8)

        if res.status_code != 200:
            print("❌ Serper HTTP Error:", res.status_code)
            return ""

        json_data = res.json()
        snippets = []

        if "answerBox" in json_data and "answer" in json_data["answerBox"]:
            snippets.append(json_data["answerBox"]["answer"])
        for item in json_data.get("organic", [])[:2]:
            if "snippet" in item:
                snippets.append(item["snippet"])

        return " ".join(snippets).strip()
    except Exception as e:
        print("❌ Serper exception:", e)
        return ""

def needs_web_context(question):
    keywords = [
        "2024", "2023", "today", "now", "current", "latest", "this year", "recent",
        "weather", "election", "result", "news", "price", "score", "exchange rate",
        "who won", "when is", "what time", "live", "president", "prime minister"
    ]
    return any(k in question.lower() for k in keywords)

def is_response_broken(text):
    banned_phrases = ["custom essay", "porn", "xxx", "retrieved from", "buy an essay"]
    return any(b in text.lower() for b in banned_phrases)

def clean_response(text):
    bad_tokens = ["user:", "question:", "q:", "note:", "recent exchange:", "[QUESTION]", "[ANSWER]"]
    for token in bad_tokens:
        if token.lower() in text.lower():
            return text.split(token)[0].strip()
    return text.strip()

def generate_zephyr_answer(context, question, history=None):
    status_message = "Generating answer..."

    if needs_web_context(question):
        web_info = get_web_summary_serper(question)
        if web_info:
            context = f"[WEB RESULT]\n{web_info.strip()}\n\n{context}"
            status_message = "Searching the internet..."
        else:
            return "⚠️ No reliable up-to-date information was found online.", "Searching the internet..."

    history_prompt = ""
    if history:
        for turn in history[-1:]:
            history_prompt += f"{turn['user']}\n{turn['bot']}\n"

    DETAIL_KEYWORDS = ["example", "give an example", "show an example", "more detail", "elaborate", "expand", "explain"]
    wants_detail = any(k in question.lower() for k in DETAIL_KEYWORDS)
    last_q = history[-1]["user"] if history else ""
    effective_question = f"{last_q} {question}" if wants_detail and len(question.split()) <= 8 else question

    style_instruction = (
        "If the user asks for more detail or an example, respond with 3–5 concise sentences. One example only."
        if wants_detail else
        "Answer in 2–3 concise sentences. Do not elaborate unless requested."
    )

    prompt = f"""
You are a helpful and concise assistant.
{style_instruction}

[CONTEXT]
{context}

[HISTORY]
{history_prompt}

[QUESTION]
{effective_question}

[ANSWER]
""".strip()

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-alpha",
            messages=[
                {"role": "system", "content": "You are a concise and factual assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )

        answer = response.choices[0].message.content.strip()
        if answer.lower().startswith("assistant:"):
            answer = answer[len("assistant:"):].strip()

        answer = clean_response(answer)

        if is_response_broken(answer):
            return "⚠️ The assistant generated an invalid response. Please try rephrasing.", status_message

        return answer, status_message

    except Exception as e:
        return f"⚠️ Error during API call: {e}", status_message
