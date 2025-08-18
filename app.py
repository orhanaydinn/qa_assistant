import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

# ==============================
# PAGE CONFIG (ilk Streamlit komutu olmak zorunda!)
# ==============================
st.set_page_config(
    page_title="AI Assistant – PDF/Image + Web + ImageGen",
    layout="wide"   # istersen "centered" de yapabilirsin
)

# ==============================
# Normal Python importları
# ==============================
import math, logging, re, time, io, base64
import numpy as np
from PIL import Image
import html as htmlmod

# ==============================
# Logging
# ==============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("app")

# ==============================
# Proje içi importlar
# (bunlar set_page_config'ten sonra gelmeli!)
# ==============================
from pdf_parser import extract_text_chunks as extract_pdf_chunks
from faiss_search import create_faiss_index as create_pdf_index
from ocr_utils import extract_ocr_chunks
from ocr_faiss import create_faiss_index as create_ocr_index
from llm_response import generate_zephyr_answer, needs_web_context
from rag_dataset_qa import load_rag_index
from embedder import embed_chunks as embed_any
from faiss_loader import load_faiss_index, get_index_status

# ==============================
# Session State Defaults
# ==============================
defaults = {
    "chat_history": [],          # [{role, content, sources?, uid?}]
    "doc_chunks": None,
    "faiss_index": None,
    "_DATASET_LOADED": False,
    "_DATASET_INDEX": None,
    "_DATASET_TEXTS": None,
    "temp_input": "",
    "pending_question": None,
    "clear_input_flag": False,
    "last_upload_type": None,    # "pdf" | "image"
    "_UPLOADED_CACHE_KEY": None,
    "_UPLOADED_NAME": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ==============================
# FAISS Index Yükleme
# ==============================
status = get_index_status()
print("FAISS status:", status)

try:
    faiss_index = load_faiss_index()   # Google Drive’dan indir & yükle
    print("✅ FAISS ready. ntotal =", getattr(faiss_index, "ntotal", 0))
except Exception as e:
    faiss_index = None
    print("❌ FAISS not available:", e)

# ==============================
# Image Generator
# ==============================
try:
    from image_gen import generate_image_from_prompt
    IMAGE_GEN_AVAILABLE = True
except Exception as e:
    IMAGE_GEN_AVAILABLE = False
    log.warning(f"image_gen not available: {e}")
# Translation (optional)
try:
    from translation_utils import smart_detect_language, translate_to_en, translate_from_en
    TRANSLATION_AVAILABLE = True
except Exception:
    TRANSLATION_AVAILABLE = False
    def smart_detect_language(x): return "en"
    def translate_to_en(x, y): return x
    def translate_from_en(x, y): return x

EN_WEB_KEYWORDS = {
    "today","yesterday","tomorrow","latest","news","weather","score",
    "match","concert","events","currency","rate","price","open now",
    "near me","traffic","tickets","holiday","reservation","euro",
    "world cup","ballon d'or","champions league","olympics","award",
    "winner","final"
}

# ==============================
# UI
# ==============================

st.markdown("""
<style>
.hero{
  margin-top:12px; 
  margin-bottom:8px; 
  text-align:center;
}

.hero h1{
  display:flex;
  align-items:center;
  gap:12px;
  white-space:nowrap;
  font-size:clamp(25px,4vw,44px);
  line-height:1.15;
  margin:0 0 4px 0;
  font-weight:300;
  letter-spacing:-0.02em;
}

.hero h1 .chip{
  display:inline-flex;
  align-items:center;
  padding:0;
  background:none;
  border-radius:0;
  font-weight:600;
  white-space:nowrap;
}

/* Dönen yazı alanı */
.rotate{
  position:relative;
  display:inline-block;
  vertical-align:middle;
  width:22ch;
  height:1.3em;
  overflow:hidden;
}

.rotate > span{
  position:absolute; left:0; top:0;
  opacity:0; transform:translateX(-40px);
  animation: wordCycle 30s ease-in-out infinite both;
  white-space:nowrap;
  color:#3b82f6;   /* 🔵 Gemini mavisi */
}

.rotate > span:nth-child(1){ animation-delay:0s; }
.rotate > span:nth-child(2){ animation-delay:6s; }
.rotate > span:nth-child(3){ animation-delay:12s; }
.rotate > span:nth-child(4){ animation-delay:18s; }
.rotate > span:nth-child(5){ animation-delay:24s; }

@keyframes wordCycle{
  0%   { opacity:0; transform:translateX(-40px); }
  8%   { opacity:1; transform:translateX(0); }
  16%  { opacity:1; transform:translateX(0); }
  20%  { opacity:0; transform:translateX(-40px);}
  100% { opacity:0; transform:translateX(-40px);}
}

.hero .muted{ 
  color:#6b7280; 
  font-size:clamp(14px,2vw,16px); 
  margin-top:6px; 
}

@media (prefers-color-scheme: dark){
  .hero .muted{ color:#9ca3af; }
}
</style>

<div class="hero">
  <h1>
    <span class="chip">Hello, I Can Do</span>
    <span class="rotate">
      <span>Answer Anything, Anytime</span>
      <span>Surf the Internet for You</span>
      <span>Dive into Any Database</span>
      <span>Understand PDFs & Images</span>
      <span>Create Stunning Images</span>
    </span>
  </h1>
  <div class="muted">Smart Chat · Documents · Web · Images</div>
</div>
""", unsafe_allow_html=True)

language_selection = st.selectbox("Select Language", ["English", "Turkish", "Auto"], key="language_option")

chat_area = st.container()
input_area = st.container()

# ==============================
# Load Dataset RAG (global küçük RAG)
# ==============================
if not st.session_state["_DATASET_LOADED"]:
    try:
        d_index, d_texts = load_rag_index()
        st.session_state["_DATASET_INDEX"] = d_index
        st.session_state["_DATASET_TEXTS"] = d_texts
        st.session_state["_DATASET_LOADED"] = True
        log.info(f"Dataset RAG loaded: {len(d_texts)} chunks")
    except Exception as e:
        log.error(f"Dataset RAG load error: {e}")

# ==============================
# Helpers
# ==============================
def has_uploaded_doc() -> bool:
    return (st.session_state.get("faiss_index") is not None) and bool(st.session_state.get("doc_chunks"))

def faiss_top_with_scores(index, chunks, q_emb, k=3):
    if index is None or not chunks:
        return [], 0.0
    if isinstance(q_emb, list):
        q_emb = np.array(q_emb)
    if len(q_emb.shape) == 1:
        q_emb = np.expand_dims(q_emb, axis=0)
    dists, idxs = index.search(q_emb, k)
    pairs, min_d = [], None
    for d, i in zip(dists[0], idxs[0]):
        if i is None or i < 0 or i >= len(chunks):
            continue
        pairs.append((chunks[i], float(d)))
        if min_d is None or d < min_d:
            min_d = float(d)
    if not pairs:
        return [], 0.0
    return pairs, math.exp(-min_d)

def build_context_filtered(chunks, question, max_chars=1600):
    if not chunks:
        return ""
    q = (question or "").lower()
    q_tokens = set(w for w in re.findall(r"[a-zA-Z0-9#\.-]+", q) if len(w) >= 3)

    m = re.search(r"\b(?:table|tab\.)\s*(\d+)\b", q)
    wanted_table_id = int(m.group(1)) if m else None

    def good_to_add(txt_len: int, total_len: int) -> bool:
        return (total_len + txt_len + 2) <= max_chars

    selected, total = [], 0
    indexed = list(enumerate(chunks))

    def is_heading(s: str) -> bool: return "[H] " in s
    def is_table(s: str) -> bool: return "[START_TABLE" in s

    def neighbor_pack(idx: int):
        nonlocal total
        h_idx = None
        j = idx
        while j >= 0:
            if is_heading(chunks[j]): h_idx = j; break
            j -= 1
        candidates = []
        if h_idx is not None: candidates.append(h_idx)
        for j in (idx-1, idx, idx+1):
            if 0 <= j < len(chunks): candidates.append(j)
        for j in candidates:
            piece = chunks[j]
            if piece in selected: continue
            if good_to_add(len(piece), total):
                selected.append(piece); total += len(piece) + 2

    if wanted_table_id is not None:
        pat = re.compile(r"\[START_TABLE[^\]]*id\s*=\s*{}\b".format(wanted_table_id))
        for idx, s in indexed:
            if pat.search(s): neighbor_pack(idx)
        if selected: return "\n\n".join(selected[:])

    wants_table_generic = ("table" in q) or ("tab." in q) or ("figure" in q) or ("fig." in q)
    if wants_table_generic:
        for idx, s in indexed:
            if not is_table(s): continue
            tok_s = set(re.findall(r"[a-zA-Z0-9#\.-]+", s.lower()))
            if q_tokens and not (q_tokens & tok_s): continue
            neighbor_pack(idx)
            if total >= int(max_chars * 0.8): break
        if selected: return "\n\n".join(selected[:])

    def is_structural(s: str) -> bool:
        s_low = s.lower()
        return ("[start_table" in s_low) or ("[list]" in s_low) or ("[h] " in s_low)

    for idx, s in indexed:
        s_low = s.lower()
        tok_s = set(re.findall(r"[a-zA-Z0-9#\.-]+", s_low))
        if q_tokens and not (q_tokens & tok_s): continue
        if is_structural(s): neighbor_pack(idx)
        if total >= int(max_chars * 0.8): break
    if selected: return "\n\n".join(selected[:])

    for s in chunks[:2]:
        if good_to_add(len(s), total):
            selected.append(s); total += len(s) + 2

    return "\n\n".join(selected[:])

def build_context_from_chunks(chunks, max_chars=1600):
    if not chunks: return ""
    text = "\n".join([c["text"] if isinstance(c, dict) and "text" in c else str(c) for c in chunks])
    return text[:max_chars]

def route_question(question, rag_confidence, doc_confidence=None):
    if has_uploaded_doc():
        return "rag_uploaded"
    q_lower = (question or "").lower()
    web_kw = any(k in q_lower for k in EN_WEB_KEYWORDS)
    try:
        web_cls = bool(needs_web_context(question))
    except Exception:
        web_cls = False
    if web_kw or web_cls:
        return "web"
    if (doc_confidence is not None and doc_confidence >= 0.44) or (rag_confidence is not None and rag_confidence >= 0.48):
        return "rag_dataset"
    return "model"

# ==============================
# Strict Retrieval + Re-ranker-lite Helpers
# ==============================
from collections import Counter
def _tokenize(text: str):
    if not text: return []
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) >= 2]

def _build_idf(chunks):
    N = max(1, len(chunks))
    df = Counter()
    for s in chunks:
        toks = set(_tokenize(s))
        df.update(toks)
    idf = {t: math.log(1.0 + N / (1.0 + df[t])) for t in df}
    return idf

def _strict_retrieve(question, chunks, faiss_index, q_emb, topn_init=40, out_top=6, allow_tables=False):
    if faiss_index is None or not chunks: return []
    if isinstance(q_emb, list): q_vec = np.array(q_emb)
    else: q_vec = q_emb
    if len(q_vec.shape) == 1: q_vec = np.expand_dims(q_vec, axis=0)

    k = min(int(topn_init), len(chunks))
    dists, idxs = faiss_index.search(q_vec, k)
    cand_idxs = [i for i in idxs[0] if i is not None and 0 <= i < len(chunks)]
    if not cand_idxs: return []

    dense_vals = [1.0/(1.0+float(dists[0][rank])) for rank,_ in enumerate(cand_idxs)]
    min_d, max_sim = (min(dense_vals), max(dense_vals)) if dense_vals else (0.0, 1.0)

    idf = _build_idf(chunks)
    q_tokens = set(_tokenize(question))
    q_idf_total = sum(idf.get(t,0.0) for t in q_tokens) + 1e-9

    scores = []
    for rank, i in enumerate(cand_idxs):
        s = chunks[i]; d = float(dists[0][rank])
        dense = 1.0/(1.0+d)
        dense = (dense - min_d) / (max_sim - min_d) if max_sim > min_d else 0.5
        toks = set(_tokenize(s))
        kw = sum(idf.get(t,0.0) for t in (q_tokens & toks)) / q_idf_total
        struct_adj = 0.0
        s_low = s.lower()
        if ("[start_table" in s_low) and (not allow_tables): struct_adj -= 0.25
        if "[h] " in s_low: struct_adj += 0.05
        if "[list]" in s_low: struct_adj += 0.03
        final = 0.6*dense + 0.4*kw + struct_adj
        scores.append((final, i))
    scores.sort(reverse=True)
    return [i for _, i in scores[:int(out_top)]]

def _strict_context(chunks, seed_indices, question, max_chars=1600, allow_tables=False):
    if not chunks or not seed_indices: return ""
    selected, total = [], 0
    def good_to_add(txt: str):
        nonlocal total
        return (total + len(txt) + 2) <= max_chars
    for idx in seed_indices:
        h_idx = None; j = idx
        while j >= 0:
            if "[H] " in chunks[j]: h_idx = j; break
            j -= 1
        neighbor_ids = []
        if h_idx is not None: neighbor_ids.append(h_idx)
        neighbor_ids.extend([idx-1, idx, idx+1])
        for j in neighbor_ids:
            if j < 0 or j >= len(chunks): continue
            piece = chunks[j]
            if (not allow_tables) and ("[START_TABLE" in piece): continue
            if piece in selected: continue
            if good_to_add(piece):
                selected.append(piece); total += len(piece) + 2
        if total >= max_chars: break
    return "\n\n".join(selected)

# ------------------------------
# Extra cleanup for "Sources:,,,," tails
# ------------------------------
def _strip_source_tails(text: str) -> str:
    """Remove dangling 'Sources:' / 'Source:' tails like 'Sources:,,,,' at the end."""
    if not text: return text
    s = text.strip()
    tail = s[-200:]
    m = re.search(r'(?is)(^|\n)\s*sources?\s*[:\-–]\s*[,;.\s·]*$', tail)
    if m:
        cut = len(s) - len(tail) + m.start()
        s = s[:cut].rstrip()
    m2 = re.search(r'(?is)(^|\n)\s*sources?\s*[:\-–]?\s*[,;.\s·]*$', s[-200:])
    if m2:
        cut = len(s) - 200 + m2.start() if len(s) > 200 else m2.start()
        s = s[:cut].rstrip()
    s = re.sub(r'[,;:\.·\u200b\ufeff\s]+$', '', s)
    return s

# --- Query türü tespiti (hava durumu mu?) ---
def _is_weather_query(q: str, body: str = "") -> bool:
    text = f"{q or ''} {body or ''}".lower()
    weather_kws = [
        "weather", "forecast", "uv", "wind", "mph", "km/h", "humidity",
        "precip", "rain", "showers", "met office", "bbc weather",
        "feels like", "°c", "°f", "gusts", "dew point"
    ]
    return any(k in text for k in weather_kws)

# ==============================
# Cleaning + Summarizing + Intent + Sentence limit
# ==============================
def _post_clean_answer(s: str, q: str = None) -> str:
    """
    Genel temizleyici: inline Source/Sources, kirik 'according to ...' ve koseli atiflari ayiklar.
    NOT: Sadece METNE uygulanir; ikon/rozet kaynaklar icin tuttugun 'sources' listesine dokunma.
    """
    if not s:
        return s
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s*\[[^\]]*\]", "", s)
    s = re.sub(r"(?im)\bSources?\s*[:\.]\s*$", "", s).strip()
    s = re.sub(r"\(\s*Sources?\s*:[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"(?im)(^|\s)[-]*\s*Sources?\s*:\s*.*?$", "", s)
    s = re.sub(r"\b(according to|provided by)\s+(,|\band\b|\s)+", "", s, flags=re.I)
    s = re.sub(r"\baccording to\s+(sources?|and|,|\s)+", "", s, flags=re.I)
    s = re.sub(r"https?://\S+\b", "", s, flags=re.I)
    s = re.sub(r"\(\s*according to[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"\(\s*source\s*:[^)]*\)", "", s, flags=re.I)
    s = re.sub(
        r"\((?:[^()]*\bhttps?://[^()]*|[^()]*\bwww\.[^()]*|[^()]*\b[a-z0-9.-]+\.(?:com|org|net|gov|co\.uk)\b[^()]*)\)",
        "", s, flags=re.I
    )
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"\(\s*,\s*\)", "", s)
    s = re.sub(r"\(\s*and\s*\)", "", s, flags=re.I)
    s = re.sub(r"\(\s*or\s*\)", "", s, flags=re.I)
    s = re.sub(r"(,\s*){2,}", ", ", s)
    s = re.sub(r",\s*and\s*,", " and ", s, flags=re.I)
    s = re.sub(r"\band\s*,\s*", "and ", s, flags=re.I)
    s = re.sub(r"\.\s*,", ".", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])([^\s])", r"\1 \2", s)
    s = re.sub(r"([.?!])([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = s.strip()                              # sadece boşlukları kırp
    s = re.sub(r"[,\u200b\ufeff\s]+$", "", s)  # sonda varsa sadece virgül/boşlukları temizle

    return s

def _detect_intent(q: str, body: str = "") -> str:
    t = f"{q or ''} {body or ''}".lower()
    if any(k in t for k in [
        "weather", "forecast", "°c", "°f", "uv index", "wind",
        "met office", "bbc weather", "accuweather", "weather.com",
        "humidity", "precip", "rain", "showers", "overcast", "sunny"
    ]): return "weather"
    if any(k in t for k in [
        "restaurant", "restaurants", "cafe", "bar", "bistro", "eat", "dine", "food",
        "pizzeria", "pizza", "best places", "top rated", "where to eat",
        "tripadvisor", "yelp", "thefork", "google reviews"
    ]): return "restaurants"
    if any(k in t for k in [
        "who is", "who are", "lecturer", "professor", "assistant professor",
        "doctor ", "dr ", "mr ", "ms ", "ceo", "cto", "founder", "author",
        "player", "footballer", "actor", "actress", "singer", "president",
        "researcher", "scientist", "instructor"
    ]): return "people"
    if any(k in t for k in [
        "news", "breaking", "headline", "latest", "announced", "reports",
        "reported by", "statement", "press release"
    ]): return "news"
    return "generic"

def _compact_summarize(text: str, q: str = "") -> str:
    """
    Cok uzun veya tekrar eden cevaplari kisaltir.
    - Weather icin: ilk 2 cumleyi tutar (temiz ve kisa bir ozet).
    - Restaurants/People/News icin: ilk satiri baslik gibi kabul eder, en fazla 3 madde ekler.
    - Generic icin: ilk 5 satir/paragraph'i tutar.
    """
    if not text: return text
    intent = _detect_intent(q, text)
    if intent == "weather":
        sents = re.split(r'(?<=[.!?])\s+', text)
        return " ".join(sents[:2]).strip()
    if intent in ("restaurants", "people", "news"):
        lines = [l.strip(" -•\t") for l in text.splitlines() if l.strip()]
        if not lines:
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
            head = sents[0] if sents else text
            rest = sents[1:4]
            bullets = ["- " + x for x in rest]
            return head + ("\n" + "\n".join(bullets) if bullets else "")
        seen, uniq = set(), []
        for l in lines:
            key = re.sub(r"\W+", "", l.lower())
            if key in seen: continue
            seen.add(key); uniq.append(l)
        head = uniq[0]
        bullets = ["- " + x for x in uniq[1:4]]
        return head + ("\n" + "\n".join(bullets) if bullets else "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines: return text
    return "\n".join(lines[:5])

INTENT_SENTENCE_LIMITS = {
    "weather": 5, "restaurants": 4, "people": 4, "news": 4, "generic": 5,
}

def _limit_sentences(text: str, q: str = "") -> str:
    if not text: return text
    intent = _detect_intent(q, text)
    max_sents = INTENT_SENTENCE_LIMITS.get(intent, 5)
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
    if not sents: return text
    seen, uniq = set(), []
    for s in sents:
        key = re.sub(r"\s+", " ", s).lower()
        if key in seen: continue
        seen.add(key); uniq.append(s)
    if len(uniq) <= max_sents: return " ".join(uniq)
    return " ".join(uniq[:max_sents])

# ==============================
# Image intent (EN + TR) ve prompt çıkarımı
# ==============================
def _detect_image_intent(q: str) -> bool:
    if not q: return False
    t = (q or "").lower()
    gen_kw = [
        "design", "draw", "illustrate", "sketch", "paint", "render",
        "generate an image", "make an image", "create an image",
        "concept art", "poster", "logo", "flyer", "icon set",
        "ui mockup", "wireframe", "3d render", "album cover",
        "infographic", "diagram", "flowchart", "meme",
        "tasarla", "çiz", "çizer misin", "illüstrasyon", "skeç", "karikatür",
        "afiş", "logo", "broşür", "el ilanı", "ikon", "ikon seti",
        "ui tasarım", "arayüz", "wireframe", "akış şeması", "diyagram",
        "meme yap", "görsel üret", "gorsel uret"
    ]
    edit_kw = [
        "edit this image", "remove background", "inpaint", "outpaint",
        "upscale", "enhance", "recolor", "replace background",
        "arka planı kaldır", "arka plan kaldır", "arka planı değiştir",
        "düzenle", "renk değiştir", "büyüt", "yükselt"
    ]
    return any(k in t for k in gen_kw + edit_kw)

def _extract_image_prompt(q: str) -> str:
    if not q: return ""
    t = q.strip()
    drop = [
        r"\b(can you|please)\b",
        r"\b(design|draw|illustrate|sketch|create|generate|render|paint)\b",
        r"\b(tasarla|çiz|çizer misin|görsel üret|gorsel uret)\b",
        r"\b(image|resim|görsel)\b"
    ]
    for d in drop:
        t = re.sub(d, "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s{2,}", " ", t).strip()
    return t or q.strip()

# ==============================
# Send message
# ==============================
def send_message():
    msg = st.session_state.temp_input.strip()
    if msg:
        st.session_state.chat_history.append({"role": "user", "content": msg})
        st.session_state.pending_question = msg

# ==============================
# Input temizleme bayrağı
# ==============================
if st.session_state.get("clear_input_flag"):
    st.session_state["temp_input"] = ""
    st.session_state["clear_input_flag"] = False

# ==============================
# Input bar
# ==============================
with input_area:
    col1, col2 = st.columns([12, 1])
    with col1:
        st.text_input(
            "Type your question...",
            key="temp_input",
            label_visibility="collapsed",
            placeholder="Type your question...",
            on_change=send_message
        )
    with col2:
        st.button("➤", on_click=send_message)
st.markdown("""
<style>
.input-container {
    text-align:center;
    margin-top:-6px;   /* input ile arayı sıklaştır */
}
.input-footer {
    text-align:center;
    font-size:12px;
    color:#6b7280;
    margin-top:0;      /* boşluk kaldırıldı */
}
.input-footer a {
    color:#6b7280;
    text-decoration:none;
}
.input-footer a:hover {
    text-decoration:underline;
}
@media (prefers-color-scheme: dark){
    .input-footer{ color:#9ca3af; }
    .input-footer a{ color:#9ca3af; }
}
</style>

<div class="input-container">
  <div class="input-footer">
    This project developed by <a href="https://www.linkedin.com/in/orhan-aydin/" target="_blank"><b>Orhan Aydin</b></a>.
  </div>
</div>
""", unsafe_allow_html=True)

# ==============================
# Sidebar: PDF / Image Uploader (tek sekmede hep açık)
# ==============================
with st.sidebar:
    uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = uploaded_file.getvalue()
        cache_key = f"{uploaded_file.name}:{len(file_bytes)}"
        is_new_file = (st.session_state.get("_UPLOADED_CACHE_KEY") != cache_key)

        if is_new_file:
            with st.spinner("Processing file..."):
                file_type = uploaded_file.type or ""
                bio = io.BytesIO(file_bytes)

                if file_type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
                    raw_chunks = extract_pdf_chunks(bio)
                    norm_chunks = []
                    for c in raw_chunks:
                        if isinstance(c, str): norm_chunks.append(c)
                        elif isinstance(c, dict) and "text" in c: norm_chunks.append(str(c["text"]))
                        elif c is not None: norm_chunks.append(str(c))
                    chunks = [c.strip() for c in norm_chunks if isinstance(c, str) and c.strip()]
                    if not chunks:
                        st.error("PDF parsed but produced no text.")
                        st.session_state["doc_chunks"] = None
                        st.session_state["faiss_index"] = None
                        st.session_state["last_upload_type"] = None
                    else:
                        emb = embed_any(chunks)
                        index = create_pdf_index(emb)
                        st.session_state["doc_chunks"] = chunks
                        st.session_state["faiss_index"] = index
                        st.session_state["last_upload_type"] = "pdf"
                else:
                    img = Image.open(bio).convert("RGB")
                    raw_chunks = extract_ocr_chunks(img)
                    chunks = []
                    for c in raw_chunks:
                        if isinstance(c, str): chunks.append(c.strip())
                        elif isinstance(c, dict) and "text" in c: chunks.append(str(c["text"]).strip())
                        elif c: chunks.append(str(c).strip())
                    chunks = [c for c in chunks if c]
                    if not chunks:
                        st.error("OCR produced no text.")
                        st.session_state["doc_chunks"] = None
                        st.session_state["faiss_index"] = None
                        st.session_state["last_upload_type"] = None
                    else:
                        emb = embed_any(chunks)
                        index = create_ocr_index(emb)
                        st.session_state["doc_chunks"] = chunks
                        st.session_state["faiss_index"] = index
                        st.session_state["last_upload_type"] = "image"

            st.session_state["_UPLOADED_CACHE_KEY"] = cache_key
            st.session_state["_UPLOADED_NAME"] = uploaded_file.name

        if st.session_state.get("doc_chunks"):
            st.info(
                f"Loaded {len(st.session_state['doc_chunks'])} doc chunks "
                f"(Source: {st.session_state.get('_UPLOADED_NAME')})"
            )
            st.success(f"File parsed and ready! ✅  (Focus: {st.session_state.get('last_upload_type')})")

# ==============================
# CSS (chat bubble + chips)
# ==============================
st.markdown("""
<style>
.chat-bubble { padding:10px 14px; border-radius:12px; margin:4px 0 6px;
  word-wrap:break-word; overflow-wrap:anywhere; white-space:pre-wrap; line-height:1.45; }
.chat-user { background:#ccecff; color:#000; margin-left:auto; text-align:left; width:fit-content; max-width:60%; }
.chat-assistant { background:#f0f0f0; color:#000; margin-right:auto; text-align:left; width:100%; }
.typing .dot { display:inline-block; opacity:0; animation:blink 1.2s infinite; }
.typing .dot:nth-child(2){animation-delay:.2s} .typing .dot:nth-child(3){animation-delay:.4s}
@keyframes blink { 0%{opacity:0} 20%{opacity:1} 60%{opacity:1} 100%{opacity:0} }
@media (max-width:640px){ .chat-user{ max-width:90% } }

.chat-assistant .src-inline { display:flex; flex-wrap:wrap; gap:6px; margin-top:12px; padding:0; line-height:1.2; }
.src-chip{ font-size:12px; background:#f3f4f6; color:#374151; padding:2px 8px; border-radius:9999px; text-decoration:none;
  white-space:nowrap; display:inline-flex; align-items:center; gap:6px; cursor:pointer; }
.src-chip:hover{ background:#e5e7eb; text-decoration:underline; }
.src-chip.link::before{ content:"🌐"; font-size:13px; line-height:1; }

.extra-sources{ display:none; margin:2px 0 0; padding:0; gap:6px; flex-wrap:wrap; }
.extra-sources .src-chip::before{ content:"🔗"; }

img.gen { max-width:100%; border-radius:12px; margin-top:8px; border:1px solid #e5e7eb; }
</style>
""", unsafe_allow_html=True)

# ==============================
# JS: source toggle
# ==============================
st.markdown("""
<script>
document.addEventListener('click', function(e){
  var el = e.target.closest('.toggle-chip');
  if(!el) return;
  var id = el.getAttribute('data-target');
  var box = document.getElementById(id);
  if(!box) return;
  var n = el.getAttribute('data-count') || '';
  var hidden = (box.style.display === '' || box.style.display === 'none');
  box.style.display = hidden ? 'flex' : 'none';
  el.textContent = (hidden ? '−' : '+') + n;
}, true);
</script>
""", unsafe_allow_html=True)

# ==============================
# Display chat history (persisted chips!)
# ==============================
with chat_area:
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-bubble chat-user'>{msg['content']}</div>",
                unsafe_allow_html=True
            )
        else:
            uid = msg.get("uid") or f"h{i}"
            content_html = _post_clean_answer(msg.get("content", ""))

            # Görsel mesajı basitçe göster (opsiyonel iyileştirme yapılabilir)
            # Eğer görsel kaydı varsa, resmi balon içinde göster
            if msg.get("image_b64"):
                caption = content_html  # ör: "Image: a good house"
                try:
                    _label, _cap = caption.split(":", 1)
                    _cap = _cap.strip()
                except ValueError:
                    _label, _cap = "Image", caption
            
                img_html = (
                    f"<div class='chat-bubble chat-assistant'>"
                    f"<div><strong>{_label}:</strong> {htmlmod.escape(_cap)}</div>"
                    f"<img class='gen' src='data:{msg.get('mime','image/png')};base64,{msg['image_b64']}' alt='generated image'/>"
                    f"</div>"
                )
                st.markdown(img_html, unsafe_allow_html=True)
                continue


            sources_hist = msg.get("sources") or []
            chips_inner = ""
            extra_html = ""
            if sources_hist:
                main_src = sources_hist[0]
                url = main_src.get("url")
                domain = (main_src.get("source") or main_src.get("title") or "Source")
                extra = len(sources_hist) - 1
                chips_inner = f"<a href='{url}' target='_blank' class='src-chip link'>{domain}</a>"
                if extra > 0:
                    chips_inner += (f"<span class='src-chip toggle-chip' data-target='extra-{uid}' data-count='{extra}'>+{extra}</span>")
                    extra_html = f"<div id='extra-{uid}' class='extra-sources' style='display:none;'>"
                    for s in sources_hist[1:]:
                        su = s.get("url")
                        sd = (s.get("source") or s.get("title") or "Source")
                        extra_html += f"<a href='{su}' target='_blank' class='src-chip link'>{sd}</a>"
                    extra_html += "</div>"

            bubble_html_hist = f"<div class='chat-bubble chat-assistant'>{content_html}"
            if chips_inner:
                bubble_html_hist += f"<div class='src-inline'>{chips_inner}</div>"
            bubble_html_hist += f"{extra_html}</div>"
            st.markdown(bubble_html_hist, unsafe_allow_html=True)

    # ==========================
    # Processing current turn
    # ==========================
    if st.session_state.pending_question:
        placeholder = st.empty()
        placeholder.markdown(
            "<div class='chat-bubble chat-assistant'>💭 Thinking "
            "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
            "</div>",
            unsafe_allow_html=True
        )

        user_input = st.session_state.pending_question
        user_lang = smart_detect_language(user_input) if TRANSLATION_AVAILABLE else "en"
        user_input_en = translate_to_en(user_input, user_lang) if (language_selection == "English" and user_lang != "en") else user_input

        # --- Image intent: design/draw -> image_gen ---
        if _detect_image_intent(user_input_en) and IMAGE_GEN_AVAILABLE:
            img_prompt = _extract_image_prompt(user_input_en) or user_input_en
            placeholder.markdown(
                "<div class='chat-bubble chat-assistant'>Creating image "
                "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
                "</div>",
                unsafe_allow_html=True
            )
            try:
                img_bytes = generate_image_from_prompt(img_prompt)
            
            # image intent bloğundaki except:
            except Exception as e:
                placeholder.markdown(
                    f"<div class='chat-bubble chat-assistant'>⚠️ Image generation failed:<br><code>{str(e)}</code></div>",
                    unsafe_allow_html=True
                )
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": f"⚠️ Image generation failed: {e}",
                    "sources": [],
                    "uid": str(int(time.time() * 1000))
                })
                st.session_state.pending_question = None
                st.session_state.clear_input_flag = True
                st.stop()


            # Success -> render and stop
            uid = str(int(time.time() * 1000))
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            
            # 1) Bu turda göster
            html_block = (
                f"<div class='chat-bubble chat-assistant'>"
                f"<div><strong>Image:</strong> {htmlmod.escape(img_prompt)}</div>"
                f"<img class='gen' src='data:image/png;base64,{b64}' alt='generated image'/>"
                f"</div>"
            )
            placeholder.markdown(html_block, unsafe_allow_html=True)
            
            # 2) Geçmişe görseli de KAYDET
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Image: {img_prompt}",
                "image_b64": b64,          # <<< önemli: base64'ü tut
                "mime": "image/png",
                "sources": [],
                "uid": uid
            })
            
            st.session_state.pending_question = None
            st.session_state.clear_input_flag = True
            st.stop()


        # ---- LLM / RAG / WEB yolu ----
        q_emb = embed_any([user_input_en])
        faiss_index_local = st.session_state.get("faiss_index")
        doc_chunks_local = st.session_state.get("doc_chunks")
        ds_index_local = st.session_state.get("_DATASET_INDEX")
        ds_texts_local = st.session_state.get("_DATASET_TEXTS")

        doc_pairs, doc_conf = faiss_top_with_scores(faiss_index_local, doc_chunks_local, q_emb, k=3)
        ds_pairs, ds_conf = faiss_top_with_scores(ds_index_local, ds_texts_local, q_emb, k=3)

        route = route_question(user_input_en, max(doc_conf, ds_conf), doc_confidence=doc_conf)

        sources = []
        if route == "rag_uploaded":
            status = "Searching on PDF " if st.session_state.get("last_upload_type") == "pdf" else "Searching on image "
            placeholder.markdown(
                f"<div class='chat-bubble chat-assistant'>{status}"
                "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
                "</div>",
                unsafe_allow_html=True
            )
            q_lower = user_input_en.lower()
            allow_tables = bool(re.search(r"\b(table|tab\.|figure|fig\.)\b", q_lower))
            seed_idxs = _strict_retrieve(user_input_en, doc_chunks_local or [], faiss_index_local, q_emb,
                                         topn_init=40, out_top=6, allow_tables=allow_tables)
            context = _strict_context(doc_chunks_local or [], seed_idxs, user_input_en, max_chars=1200,
                                      allow_tables=allow_tables)
            if not context:
                top_chunks = [c for c, _ in (doc_pairs or [])] or (doc_chunks_local or [])[:10]
                context = build_context_filtered(top_chunks, user_input_en, max_chars=1200)
            answer_text, sources = generate_zephyr_answer(context, user_input_en,
                                                          st.session_state.get("chat_history", []),
                                                          force_web=False)

        elif route == "rag_dataset":
            placeholder.markdown(
                "<div class='chat-bubble chat-assistant'>Searching on database "
                "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
                "</div>",
                unsafe_allow_html=True
            )
            q_lower = user_input_en.lower()
            allow_tables = bool(re.search(r"\b(table|tab\.|figure|fig\.)\b", q_lower))
            seed_idxs = _strict_retrieve(user_input_en, ds_texts_local or [], ds_index_local, q_emb,
                                         topn_init=40, out_top=6, allow_tables=allow_tables)
            context = _strict_context(ds_texts_local or [], seed_idxs, user_input_en, max_chars=1200,
                                      allow_tables=allow_tables)
            if not context:
                top_chunks = [c for c, _ in (ds_pairs or [])]
                context = build_context_filtered(top_chunks, user_input_en, max_chars=1200)
            answer_text, sources = generate_zephyr_answer(context, user_input_en,
                                                          st.session_state.get("chat_history", []),
                                                          force_web=False)

        elif route == "web":
            placeholder.markdown(
                "<div class='chat-bubble chat-assistant'>Searching on internet "
                "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
                "</div>",
                unsafe_allow_html=True
            )
            answer_text, sources = generate_zephyr_answer("", user_input_en,
                                                          st.session_state.get("chat_history", []),
                                                          force_web=True)
        else:
            placeholder.markdown(
                "<div class='chat-bubble chat-assistant'>Thinking "
                "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
                "</div>",
                unsafe_allow_html=True
            )
            answer_text, sources = generate_zephyr_answer("", user_input_en,
                                                          st.session_state.get("chat_history", []),
                                                          force_web=False)

        # ----- Post-process: temizle → özetle → cümle limiti -----
        answer_text_clean = _post_clean_answer(answer_text or "", q=user_input_en)
        answer_text_clean = _compact_summarize(answer_text_clean, q=user_input_en)
        answer_text_clean = _limit_sentences(answer_text_clean, q=user_input_en)

        if language_selection == "Turkish":
            answer_text_clean = translate_from_en(answer_text_clean, "tr")

        uid = str(int(time.time() * 1000))  # toggle için benzersiz id

        # chip'ler (balon içinde) — +N tıklanabilir
        chips_inner = ""
        extra_html = ""
        if sources:
            main_src = sources[0]
            url = main_src.get("url")
            domain = (main_src.get("source") or main_src.get("title") or "Source")
            extra = len(sources) - 1

            chips_inner = f"<a href='{url}' target='_blank' class='src-chip link'>{domain}</a>"
            if extra > 0:
                chips_inner += (
                    f"<span class='src-chip toggle-chip' data-target='extra-{uid}' data-count='{extra}'>+{extra}</span>"
                )
                extra_html = f"<div id='extra-{uid}' class='extra-sources' style='display:none;'>"
                for s in sources[1:]:
                    su = s.get("url")
                    sd = (s.get("source") or s.get("title") or "Source")
                    extra_html += f"<a href='{su}' target='_blank' class='src-chip link'>{sd}</a>"
                extra_html += "</div>"

        # Balonu yaz
        bubble_html = f"<div class='chat-bubble chat-assistant'>{answer_text_clean}"
        if chips_inner:
            bubble_html += f"<div class='src-inline'>{chips_inner}</div>"
        bubble_html += f"{extra_html}</div>"

        placeholder.markdown(bubble_html, unsafe_allow_html=True)

        # History'ye kaydet
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer_text_clean,
            "sources": sources or [],
            "uid": uid
        })
        st.session_state.pending_question = None
        st.session_state.clear_input_flag = True
