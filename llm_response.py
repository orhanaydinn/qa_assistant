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
# Web bağlamı gerekir mi? (TEK KAYNAK)
# -------------------------------------------------
def needs_web_context(question: str) -> bool:
    """
    Güncel/harici bilgi gerektiren soruları yakalar.
    Tek karar kaynağı: llm_response.needs_web_context
    """
    q_raw = _safe_str(question)
    q = q_raw.lower().strip()
    if not q:
        return False

    # 0) Açık web isteği (override)
    explicit = [
        "look up", "search the web", "check the web", "google it", "bing it",
        "kaynağına bak", "internete bak", "webde ara", "google'da ara",
        "doğrula", "verify from sources", "kaynak ver", "kaynakları göster",
        "link", "bağlantı at", "kaynak paylaş"
    ]
    if any(k in q for k in explicit):
        return True

    # 1) Haber/güncellik/sürüm/saha
    news = {
        "news","headline","breaking","latest","report","press release","announcement",
        "election","elections","vote","poll","turnout","protest","strike","verdict",
        "launch","release date","patch notes","update rolled out",
        "haber","son dakika","güncel","açıklama","duyuru","basın açıklaması",
        "seçim","oy oranı","anket","katılım","protesto","grev","mahkeme kararı",
        "çıkış tarihi","güncelleme notları","güncelleme yayımlandı"
    }
    sports = {
        "score","scores","match","game","fixture","fixtures","results","table","standings",
        "champions league","premier league","nba","nfl","nhl","mlb","epl","uefa","fifa","olympics",
        "world cup","final","semi-final","quarter-final","cup draw","transfer",
        "skor","maç","sonuç","fikstür","puan durumu","transfer","yarı final","çeyrek final","kupa"
    }
    finance = {
        "price","prices","stock","stocks","share price","earnings","guidance","ipo","market cap",
        "bitcoin","btc","ethereum","eth","crypto","exchange rate","usd/try","eur/try","usd try","eur try",
        "inflation","cpi","interest rate","fed","ecb","bond yield",
        "fiyat","fiyatlar","hisse","borsa","kazanç","halka arz","piyasa değeri",
        "döviz","kur","enflasyon","faiz","merkez bankası","tahvil faizi"
    }
    local = {
        "weather","forecast","uv index","wind","humidity","precip","rain",
        "near me","open now","traffic","accident","road closed","bus times","train times",
        "restaurant","cafe","bar","concert","festival","event","tickets",
        "hava","hava durumu","tahmin","uv","rüzgar","nem","yağış","yağmur",
        "yakınımda","şimdi açık","trafik","kaza","yol kapalı","otobüs saatleri","tren saatleri",
        "etkinlik","konser","bilet","restoran","kafe","bar"
    }
    tech = {
        "release notes","release date","changelog","what's new","firmware","driver update",
        "security update","cve-","zero-day","exploit","patch tuesday",
        "sürüm notları","güncelleme notları","yenilikler","bellenim","sürücü güncellemesi",
        "güvenlik güncellemesi","güvenlik açığı","yama"
    }

    if any(k in q for k in (news | sports | finance | local | tech)):
        return True

    # 2) Güvenli regex kalıpları (UTF-8 karakterlerle çakışma ihtimalini azaltmak için)
    # - Para birimleri: \$, \u20AC (Euro), \u00A3 (Pound)
    # - Tarih/yıl/skor/parite/sürüm/CVE
    regex_strings = [
        r"\b20(2[3-9]|[3-9]\d)\b",                              # 2023+
        r"\b(0??\d{2}\b",  # dd/mm/yyyy
        r"\b(0??\d{2}\b",  # mm/dd/yyyy
        r"\b(0?[1-9]|[12]\d|3[01])\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
        r"\b(0?[1-9]|[12]\d|3[01])\s+(ocak|şub|mart|nis|may|haz|tem|ağu|eyl|ekim|kas|ara)\b",
        r"\b(at|saat)\s*\d{1,2}(:\d{2})?\s*(am|pm)?\b",
        r"\b\d{1,2}\s*[-–:]\s*\d{1,2}\b",                       # skor: 2-1, 1:0
        r"\b([\$\u20AC\u00A3])\s?\d{1,3}(,\d{3})*(\.\d+)?\b",   # $ 1,234.56 / €99 / £10
        r"\b\d+(?:[.,]\d+)?\s*(tl|usd|eur|gbp)\b",
        r"\b([A-Z]{3})/([A-Z]{3})\b",                           # USD/TRY
        r"\b[A-Z]{2,5}\s+price\b",                              # TICKER price
        r"\bv?\d+\.\d+(\.\d+)?\b",                              # v1.2.3
        r"\bcve-\d{4}-\d{4,}\b",                                # CVE-2024-12345
    ]

    import re
    for pat in regex_strings:
        try:
            if re.search(pat, q, flags=re.I):
                return True
        except re.error:
            # Geçersiz bir pattern varsa sessizce atla (ortam/encoding farkı olabilir)
            continue

    # 3) Göreli zamanlar
    rel = ["bugün", "yarın", "dün", "this morning", "tonight", "tomorrow", "yesterday", "right now", "az önce"]
    if any(k in q for k in rel):
        return True

    return False


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
        sources = search_serper_structured(question, num=8, tbs=None)
        if sources:
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

    DETAIL_KWS = ["example", "give an example", "show an example", "more detail", "elaborate", "expand", "explain",
                  "örnek", "detay", "açıkla"]
    wants_detail = any(k in (_safe_str(question)).lower() for k in DETAIL_KWS)
    style_instruction = (
        "If the user asks for more detail or an example, respond with 3–5 concise sentences. One example only."
        if wants_detail else
        "Answer in 3–6 concise sentences. Be direct."
    )

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
            temperature=0.3,
            max_tokens=400
        )
        answer = (_safe_str(resp.choices[0].message.content)).strip()
        if answer.lower().startswith("assistant:"):
            answer = answer[len("assistant:"):].strip()

        answer = clean_response(answer)
        if is_response_broken(answer):
            answer = "The assistant generated an invalid response. Please try rephrasing."

        return (answer or "I couldn't generate a response."), (sources or [])

    except Exception as e:
        return f"Error during API call: {e}", []
