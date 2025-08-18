# -*- coding: utf-8 -*-
"""
Updated on Sat Aug 16 2025

@author: Orhan
"""

# llm_response.py
import re
import requests
import streamlit as st
from urllib.parse import urlparse
from datetime import datetime
from huggingface_hub import InferenceClient

# ---- Secrets / Keys ----
HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
SERPER_API_KEY = st.secrets.get("SERPER_API_KEY", "")

# Zephyr (alpha) – Inference API üzerinden
client = InferenceClient(model="HuggingFaceH4/zephyr-7b-beta", token=HF_API_TOKEN)

# -------------------------------------------------
# Yardımcılar
# -------------------------------------------------
def _normalize_domain(u: str) -> str:
    try:
        d = urlparse(u).netloc.lower()
        return d[4:] if d.startswith("www.") else d
    except Exception:
        return ""

def _safe_str(x) -> str:
    return x if isinstance(x, str) else ("" if x is None else str(x))

# -------------------------------------------------
# Web arama (Serper) — Eski basit özet (geri-uyumluluk için)
# -------------------------------------------------
def get_web_summary_serper(query: str) -> str:
    """DEPRECATION: Yapılandırılmış arama yerine kullanmayın; geriye uyumluluk için tutuldu."""
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

# -------------------------------------------------
# Web arama (Serper) — Yapılandırılmış sonuçlar
# -------------------------------------------------
def search_serper_structured(query: str, num: int = 10, tbs: str | None = None):
    """
    SERPER'den title/url/date/domain/snippet şeklinde yapılandırılmış sonuç döndürür.
    Dönüş: list[ {title, url, snippet, date, source} ]
    """
    if not SERPER_API_KEY:
        return []

    headers = {"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"}
    data = {"q": query, "num": num}
    if tbs:
        data["tbs"] = tbs  # e.g., 'qdr:m' (past month), 'qdr:y' (past year)

    try:
        r = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=8)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        print(f"[Serper structured error] {e}")
        return []

    items = []

    # AnswerBox (varsa)
    ab = js.get("answerBox") or {}
    if isinstance(ab, dict) and (ab.get("snippet") or ab.get("answer")):
        items.append({
            "title": _safe_str(ab.get("title") or "Answer Box"),
            "url": _safe_str(ab.get("link") or ""),
            "snippet": _safe_str(ab.get("answer") or ab.get("snippet") or ""),
            "date": _safe_str(ab.get("date") or ab.get("dateUtc") or ""),
            "source": _normalize_domain(_safe_str(ab.get("link") or "")),
        })

    # Organic sonuçlar
    for it in js.get("organic", []) or []:
        u = _safe_str(it.get("link") or "")
        items.append({
            "title": _safe_str(it.get("title") or ""),
            "url": u,
            "snippet": _safe_str(it.get("snippet") or ""),
            "date": _safe_str(it.get("date") or it.get("dateUtc") or ""),
            "source": _normalize_domain(u),
        })

    # Basit çeşitlilik: domain başına en fazla 2 sonuç
    seen_per_domain = {}
    diverse = []
    for x in items:
        d = x["source"]
        c = seen_per_domain.get(d, 0)
        if c < 2:
            diverse.append(x)
            seen_per_domain[d] = c + 1

    return diverse[:num]

# -------------------------------------------------
# Web bağlamı gerekir mi?
# -------------------------------------------------
def needs_web_context(question: str) -> bool:
    # EN + TR anahtar kelimeler
    kws = [
        # yıllar / güncellik
        "2026","2025","2024","2023","today","now","current","latest","recent",
        "bugün","güncel","son","şimdi","az önce",
        # haber/sonuç/fiyat
        "news","headline","result","score","live","price","exchange rate","usd try","eur try",
        "haber","sonuç","skor","canlı","fiyat","kur","döviz",
        # zaman/saat/schedule
        "when is","what time","schedule","ne zaman","saat kaçta","program",
        # spor / seçim vb.
        "who won","champion","final","president","prime minister","seçim","maç","finali","kazanan",
        # yerel bilgi
        "weather","hava durumu","trafik","yakınımda","near me","restaurant","etkinlik"
    ]
    q = (_safe_str(question)).lower()
    return any(k in q for k in kws)

# -------------------------------------------------
# Temizlik / güvenlik
# -------------------------------------------------
def is_response_broken(text: str) -> bool:
    banned = ["custom essay", "porn", "xxx", "retrieved from", "buy an essay"]
    t = (_safe_str(text)).lower()
    return any(b in t for b in banned)

def clean_response(text: str) -> str:
    if not isinstance(text, str):
        text = str(text or "")
    bad_tokens = [
        "user:", "question:", "q:", "note:", "recent exchange:",
        "[QUESTION]", "[ANSWER]", "[CORRECTION]", "[HISTORY]", "[/HISTORY]",
        "<|>", "[/RESULT]", "[INST]", "[/INST]", "[/USER] " "[/ASSISTANT]",
        "[USER]", "[ASSISTANT]", "[QUERY]", "[WEB RESULT]", "[CONTEXT]", "[IMAGE]", "[SCAN]", "[/USER]"
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

# -------------------------------------------------
# History'yi tek bir metne dönüştür
# -------------------------------------------------
def _history_to_text(history) -> str:
    """
    Desteklenen formatlar:
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
    return "\n".join(buf).strip()

# -------------------------------------------------
# LLM cevabı üret — (cevap, kaynaklar) döner
# -------------------------------------------------
def generate_zephyr_answer(context: str, question: str, history=None, preview: bool=False, force_web: bool=False):
    """
    DÖNÜŞ:
      answer_text: str
      sources: list[{title,url,snippet,date,source}]
    """
    sources = []

    # Web gerekiyorsa: yapılandırılmış sonuçları getir ve LLM'e [SOURCES] ver
    if force_web or needs_web_context(question):
        # Not: tbs='qdr:y' → “past year”, ihtiyaca göre 'qdr:m' da verilebilir.
        sources = search_serper_structured(question, num=8, tbs=None)
        if sources:
            # LLM'e verilecek numaralanmış kaynak bloğu
            numbered = []
            for i, s in enumerate(sources, 1):
                line = f"[{i}] {s['title']} — {s['source']} — {s['url']}"
                if s.get("date"):
                    line += f" — {s['date']}"
                if s.get("snippet"):
                    line += f"\nSummary: {s['snippet']}"
                numbered.append(line)
            context = (context or "")
            context = f"{context}\n\n[SOURCES]\n" + "\n\n".join(numbered)

    history_prompt = _history_to_text(history)

    # Kullanıcı detay istiyorsa üslubu ayarla
    DETAIL_KWS = ["example", "give an example", "show an example", "more detail", "elaborate", "expand", "explain",
                  "örnek", "detay", "açıkla"]
    wants_detail = any(k in (_safe_str(question)).lower() for k in DETAIL_KWS)
    style_instruction = (
        "If the user asks for more detail or an example, respond with 3–5 concise sentences. One example only."
        if wants_detail else
        "Answer in 3–6 concise sentences. Be direct."
    )

    # Atıflı, kaynak zorunlu prompt
    prompt = f"""
You are a factual assistant. Prefer recent and reputable sources.

Rules:
- Use ONLY the information from [SOURCES] and [CONTEXT].
- Add citation markers like [1], [2] immediately after the claims they support.
- If the answer is not present in the sources/context, say "I don't know."
- Keep it concise. {style_instruction}

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
                {"role": "system", "content": "You are a concise, factual assistant who always cites sources like [1], [2]."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,          # daha tutarlı atıf için düşük sıcaklık
            max_tokens=400
        )
        answer = (_safe_str(resp.choices[0].message.content)).strip()
        if answer.lower().startswith("assistant:"):
            answer = answer[len("assistant:"):].strip()

        answer = clean_response(answer)
        if is_response_broken(answer):
            answer = "The assistant generated an invalid response. Please try rephrasing."

        # Dönüş: cevap + kaynak listesi
        return (answer or "I couldn't generate a response."), (sources or [])

    except Exception as e:
        return f"Error during API call: {e}", []
