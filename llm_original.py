# -*- coding: utf-8 -*-
"""
Created on Thu Aug 14 15:29:09 2025

@author: Orhan
"""

# llm_response.py
import requests
import streamlit as st
from huggingface_hub import InferenceClient

# ---- Secrets / Keys ----
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "")

# Zephyr (alpha) – Inference API üzerinden
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_API_TOKEN)


# ---------------------------
# Web arama (Serper)
# ---------------------------
def get_web_summary_serper(query: str) -> str:
    if not SERPER_API_KEY:
        return ""
    try:
        headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
        data = {"q": query}
        res = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=8)
        res.raise_for_status()
        js = res.json()

        parts = []
        # answerBox varsa önce onu koy
        ab = js.get("answerBox", {})
        if isinstance(ab, dict):
            ans = ab.get("answer") or ab.get("snippet")
            if ans:
                parts.append(str(ans))

        # kısa, güvenli snippetler
        for item in js.get("organic", [])[:2]:
            snip = item.get("snippet")
            if snip:
                parts.append(snip)

        return " ".join(parts).strip()
    except Exception as e:
        print(f"[Serper error] {e}")
        return ""


# ---------------------------
# Web bağlamı gerekir mi?
# ---------------------------
def needs_web_context(question: str) -> bool:
    kws = [
        "2025", "2024", "today", "now", "current", "latest", "recent",
        "weather", "election", "result", "news", "price", "score",
        "exchange rate", "who won", "when is", "what time", "live",
        "president", "prime minister", "best", "restaurant"
    ]
    q = (question or "").lower()
    return any(k in q for k in kws)


# ---------------------------
# Temizlik / güvenlik
# ---------------------------
def is_response_broken(text: str) -> bool:
    banned = ["custom essay", "porn", "xxx", "retrieved from", "buy an essay"]
    t = (text or "").lower()
    return any(b in t for b in banned)

def clean_response(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    bad_tokens = [
        "user:", "question:", "q:", "note:", "recent exchange:",
        "[QUESTION]", "[ANSWER]", "[CORRECTION]", "[HISTORY]", "[/HISTORY]",
        "<|>", "[/RESULT]", "[INST]", "[/INST]", "[/USER] " "[/ASSISTANT]",
        "[/USER]", "[ASSISTANT]", "[QUERY]", "[WEB RESULT]", "[CONTEXT]", "[IMAGE]", "[SCAN]"
    ]
    low = text.lower()
    cut = None
    for tok in bad_tokens:
        i = low.find(tok.lower())
        if i != -1:
            cut = i if cut is None else min(cut, i)
    if cut is not None:
        text = text[:cut]
    return text.strip()


# ---------------------------
# History'yi tek bir metne dönüştür
# ---------------------------
def _history_to_text(history) -> str:
    """
    Şu formatların hepsini destekler:
      - [{"role": "user"/"assistant", "content": "..."}]
      - [{"user": "...", "bot": "..."}]
      - [("user mesajı", "assistant cevabı"), ...]
    Sondan en fazla 3 tur eklenir.
    """
    if not history:
        return ""
    buf = []
    last_turns = history[-3:] if len(history) > 3 else history
    for turn in last_turns:
        if isinstance(turn, dict):
            if "role" in turn and "content" in turn:
                role = turn["role"]
                content = turn["content"]
                if role == "user":
                    buf.append(f"User: {content}")
                elif role == "assistant":
                    buf.append(f"Assistant: {content}")
            else:
                u = turn.get("user", "")
                b = turn.get("bot", "")
                if u:
                    buf.append(f"User: {u}")
                if b:
                    buf.append(f"Assistant: {b}")
        elif isinstance(turn, (list, tuple)) and len(turn) == 2:
            u, b = turn
            if u:
                buf.append(f"User: {u}")
            if b:
                buf.append(f"Assistant: {b}")
        # farklı bir tipse geç
    return "\n".join(buf).strip()


# ---------------------------
# LLM cevabı üret (sadece str döner)
# ---------------------------
def generate_zephyr_answer(context: str, question: str, history=None, preview: bool=False, force_web: bool=False) -> str:
    # Web gerekiyorsa özet ekle
    if force_web or needs_web_context(question):
        web_info = get_web_summary_serper(question)
        if web_info:
            context = f"[WEB RESULT]\n{web_info}\n\n{context}"

    history_prompt = _history_to_text(history)

    # Kullanıcı detay istiyorsa üslubu ayarla
    DETAIL_KWS = ["example", "give an example", "show an example", "more detail", "elaborate", "expand", "explain"]
    wants_detail = any(k in (question or "").lower() for k in DETAIL_KWS)
    style_instruction = (
        "If the user asks for more detail or an example, respond with 3–5 concise sentences. One example only."
        if wants_detail else
        "Answer in 2–3 concise sentences. Do not elaborate unless requested."
    )

    # Stabil prompt
    prompt = f"""
You are a helpful and concise assistant helping users extract factual information from receipts, documents, or images.

Always answer only using the information provided in the [CONTEXT]. Do not hallucinate or assume.

If the context contains direct numerical values (e.g., "Sales tax: $8.55"), extract them directly.

If the answer is not present, reply: "I don't know."
{style_instruction}

[CONTEXT]
{context or ""}

[HISTORY]
{history_prompt or ""}

[QUESTION]
{question}

[ANSWER]
""".strip()

    try:
        resp = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-beta",
            messages=[
                {"role": "system", "content": "You are a concise and factual assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=300
        )
        answer = (resp.choices[0].message.content or "").strip()
        if answer.lower().startswith("assistant:"):
            answer = answer[len("assistant:"):].strip()

        answer = clean_response(answer)
        if is_response_broken(answer):
            return "The assistant generated an invalid response. Please try rephrasing."
        return answer or "I couldn't generate a response."
    except Exception as e:
        return f"Error during API call: {e}"
