# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 16:09:52 2025

@author: Orhan
"""

import re, time, base64, html as htmlmod
import numpy as np
import streamlit as st

# Projendeki mevcut modüller
from embedder import embed_chunks as embed_any
from rag_dataset_qa import load_rag_index
from llm_response import generate_zephyr_answer, needs_web_context  # <-- tek kaynak

# UI yardımcıları
from main_ui import render_sources_chips

# (opsiyonel) görsel üretim
try:
    from image_gen import generate_image_from_prompt
    IMAGE_GEN_AVAILABLE = True
except Exception:
    IMAGE_GEN_AVAILABLE = False

# (opsiyonel) çeviri
try:
    from translation_utils import smart_detect_language, translate_to_en, translate_from_en
    TRANSLATION_AVAILABLE = True
except Exception:
    TRANSLATION_AVAILABLE = False
    def smart_detect_language(x): return "en"
    def translate_to_en(x, y): return x
    def translate_from_en(x, y): return x

# --- intent/temizlik & retrieval yardımcıları ---

INTENT_SENTENCE_LIMITS = {"weather": 5, "restaurants": 4, "people": 4, "news": 4, "generic": 5}

def _detect_intent(q: str, body: str = "") -> str:
    t = f"{q or ''} {body or ''}".lower()
    if any(k in t for k in ["weather", "forecast", "°c", "°f", "uv index", "wind", "humidity", "precip", "rain", "sunny"]):
        return "weather"
    if any(k in t for k in ["restaurant", "restaurants", "cafe", "bar", "bistro", "eat", "dine", "food", "pizza"]):
        return "restaurants"
    if any(k in t for k in ["who is", "ceo", "founder", "author", "player", "actor", "singer", "president"]):
        return "people"
    if any(k in t for k in ["news", "breaking", "latest", "announced", "reports", "press release"]):
        return "news"
    return "generic"

def _post_clean_answer(s: str) -> str:
    if not s: return s
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = re.sub(r"\s*\[[^\]]*\]", "", s)
    s = re.sub(r"(?im)\bSources?\s*[:\.]\s*$", "", s).strip()
    s = re.sub(r"\(\s*according to[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"\(\s*source\s*:[^)]*\)", "", s, flags=re.I)
    s = re.sub(r"https?://\S+\b", "", s, flags=re.I)
    s = re.sub(r"\(\s*\)", "", s)
    s = re.sub(r"(,\s*){2,}", ", ", s)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"([,;:])([^\s])", r"\1 \2", s)
    s = re.sub(r"([.?!])([A-Za-z])", r"\1 \2", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _compact_summarize(text: str, q: str = "") -> str:
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
        head = uniq[0]; bullets = ["- " + x for x in uniq[1:4]]
        return head + ("\n" + "\n".join(bullets) if bullets else "")
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return "\n".join(lines[:5]) if lines else text

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
    return " ".join(uniq[:max_sents])

def _has_uploaded_doc() -> bool:
    return (st.session_state.get("faiss_index") is not None) and bool(st.session_state.get("doc_chunks"))

def _faiss_top_with_scores(index, chunks, q_emb, k: int = 3):
    if index is None or not chunks: return [], 0.0
    if isinstance(q_emb, list): q_emb = np.array(q_emb)
    if len(q_emb.shape) == 1: q_emb = np.expand_dims(q_emb, axis=0)
    dists, idxs = index.search(q_emb, k)
    pairs, min_d = [], None
    for d, i in zip(dists[0], idxs[0]):
        if i is None or i < 0 or i >= len(chunks): continue
        pairs.append((chunks[i], float(d)))
        if min_d is None or d < min_d: min_d = float(d)
    if not pairs: return [], 0.0
    import math
    return pairs, float(np.exp(-min_d))

def _strict_retrieve(question: str, chunks, faiss_index, q_emb, topn_init=40, out_top=6, allow_tables=False):
    """FAISS + anahtar kelime ağırlığı ile en iyi komşuları seç."""
    if faiss_index is None or not chunks: return []
    if isinstance(q_emb, list): q_vec = np.array(q_emb)
    else: q_vec = q_emb
    if len(q_vec.shape) == 1: q_vec = np.expand_dims(q_vec, axis=0)

    k = min(int(topn_init), len(chunks))
    dists, idxs = faiss_index.search(q_vec, k)
    cand = [i for i in idxs[0] if i is not None and 0 <= i < len(chunks)]
    if not cand: return []

    # IDF hesapla
    from collections import Counter
    def _tok(t): return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if len(w) >= 2]
    N = max(1, len(chunks)); df = Counter()
    for s in chunks: df.update(set(_tok(s)))
    import math
    idf = {t: math.log(1.0 + N / (1.0 + df[t])) for t in df}
    q_tokens = set(_tok(question))
    q_idf_total = sum(idf.get(t, 0.0) for t in q_tokens) + 1e-9

    dense_vals = [1.0/(1.0+float(dists[0][rank])) for rank,_ in enumerate(cand)]
    mn, mx = (min(dense_vals), max(dense_vals)) if dense_vals else (0.0, 1.0)

    scores = []
    for rank, i in enumerate(cand):
        s = chunks[i]; d = float(dists[0][rank])
        dense = 1.0/(1.0+d)
        dense = (dense - mn) / (mx - mn) if mx > mn else 0.5
        toks = set(_tok(s))
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

def _strict_context(chunks, seed_indices, question, max_chars=1200, allow_tables=False):
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

def _build_context_fallback(chunks, question, max_chars=1200):
    if not chunks: return ""
    q = (question or "").lower()
    q_tokens = set(w for w in re.findall(r"[a-zA-Z0-9#\.-]+", q) if len(w) >= 3)
    selected, total = [], 0
    def good_to_add(tlen): 
        nonlocal total
        return (total + tlen + 2) <= max_chars
    for s in chunks:
        tok_s = set(re.findall(r"[a-zA-Z0-9#\.-]+", s.lower()))
        if q_tokens and not (q_tokens & tok_s): continue
        if good_to_add(len(s)):
            selected.append(s); total += len(s) + 2
        if total >= int(max_chars*0.8): break
    if not selected:
        for s in chunks[:2]:
            if good_to_add(len(s)): selected.append(s); total += len(s) + 2
    return "\n\n".join(selected)

def _route_question(question: str, rag_confidence: float, doc_confidence: float = None) -> str:
    if _has_uploaded_doc():
        return "rag_uploaded"
    try:
        web_cls = bool(needs_web_context(question))
    except Exception:
        web_cls = False
    if web_cls:
        return "web"
    if (doc_confidence is not None and doc_confidence >= 0.44) or (rag_confidence is not None and rag_confidence >= 0.48):
        return "rag_dataset"
    return "model"

# --- public API ---

def preload_small_rag():
    if not st.session_state.get("_DATASET_LOADED"):
        try:
            d_index, d_texts = load_rag_index()
            st.session_state.update({
                "_DATASET_INDEX": d_index,
                "_DATASET_TEXTS": d_texts,
                "_DATASET_LOADED": True,
            })
        except Exception as e:
            st.error(f"Dataset RAG load error: {e}")

def handle_user_message(user_input: str, language_selection: str):
    placeholder = st.empty()
    placeholder.markdown(
        "<div class='chat-bubble chat-assistant'>Thinking "
        "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Dil yönlendirme
    user_lang = smart_detect_language(user_input) if TRANSLATION_AVAILABLE else "en"
    user_input_en = translate_to_en(user_input, user_lang) if (language_selection == "English" and user_lang != "en") else user_input

    # Görsel niyet (varsa önce)
    def _detect_image_intent(q: str) -> bool:
        if not q: return False
        t = q.lower()
        gen_kw = ["design","draw","illustrate","sketch","paint","render","generate an image","make an image",
                  "concept art","poster","logo","flyer","icon","ui mockup","wireframe","3d render","infographic",
                  "diagram","flowchart","meme","tasarla","çiz","illüstrasyon","karikatür","afiş","broşür","ikon",
                  "ui tasarım","arayüz","akış şeması","görsel üret","gorsel uret"]
        edit_kw = ["remove background","inpaint","outpaint","upscale","enhance","recolor",
                   "arka plan","düzenle","renk değiştir","büyüt","yükselt"]
        return any(k in t for k in gen_kw + edit_kw)

    def _extract_image_prompt(q: str) -> str:
        t = q.strip()
        for pat in [r"\b(can you|please)\b", r"\b(design|draw|illustrate|sketch|create|generate|render|paint)\b",
                    r"\b(tasarla|çiz|çizer misin|görsel üret|gorsel uret)\b", r"\b(image|resim|görsel)\b"]:
            t = re.sub(pat, "", t, flags=re.I)
        return re.sub(r"\s{2,}", " ", t).strip() or q.strip()

    if IMAGE_GEN_AVAILABLE and _detect_image_intent(user_input_en):
        img_prompt = _extract_image_prompt(user_input_en)
        placeholder.markdown(
            "<div class='chat-bubble chat-assistant'>Creating image "
            "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
            "</div>",
            unsafe_allow_html=True,
        )
        try:
            img_bytes = generate_image_from_prompt(img_prompt)
        except Exception as e:
            placeholder.markdown(
                f"<div class='chat-bubble chat-assistant'>Image generation failed:<br><code>{htmlmod.escape(str(e))}</code></div>",
                unsafe_allow_html=True,
            )
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Image generation failed: {e}",
                "sources": [],
                "uid": str(int(time.time() * 1000)),
            })
            return

        uid = str(int(time.time() * 1000))
        b64 = base64.b64encode(img_bytes).decode("utf-8")
        html_block = (
            f"<div class='chat-bubble chat-assistant'>"
            f"<div><strong>Image:</strong> {htmlmod.escape(img_prompt)}</div>"
            f"<img class='gen' src='data:image/png;base64,{b64}' alt='generated image'/>"
            f"</div>"
        )
        placeholder.markdown(html_block, unsafe_allow_html=True)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": f"Image: {img_prompt}",
            "image_b64": b64,
            "mime": "image/png",
            "sources": [],
            "uid": uid,
        })
        return

    # Embeddings & mevcut indexler
    q_emb = embed_any([user_input_en])
    faiss_index_local = st.session_state.get("faiss_index")
    doc_chunks_local = st.session_state.get("doc_chunks")
    ds_index_local = st.session_state.get("_DATASET_INDEX")
    ds_texts_local = st.session_state.get("_DATASET_TEXTS")

    doc_pairs, doc_conf = _faiss_top_with_scores(faiss_index_local, doc_chunks_local, q_emb, k=3)
    ds_pairs, ds_conf = _faiss_top_with_scores(ds_index_local, ds_texts_local, q_emb, k=3)
    route = _route_question(user_input_en, max(doc_conf, ds_conf), doc_confidence=doc_conf)

    sources = []
    if route == "rag_uploaded":
        placeholder.markdown(
            "<div class='chat-bubble chat-assistant'>Searching on document "
            "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
            "</div>",
            unsafe_allow_html=True,
        )
        allow_tables = bool(re.search(r"\b(table|tab\.|figure|fig\.)\b", user_input_en.lower()))
        seed_idxs = _strict_retrieve(user_input_en, doc_chunks_local or [], faiss_index_local, q_emb,
                                     topn_init=40, out_top=6, allow_tables=allow_tables)
        context = _strict_context(doc_chunks_local or [], seed_idxs, user_input_en, max_chars=1200,
                                  allow_tables=allow_tables)
        if not context:
            top_chunks = [c for c, _ in (doc_pairs or [])] or (doc_chunks_local or [])[:10]
            context = _build_context_fallback(top_chunks, user_input_en, max_chars=1200)
        answer_text, sources = generate_zephyr_answer(context, user_input_en, st.session_state.get("chat_history", []), force_web=False)

    elif route == "rag_dataset":
        placeholder.markdown(
            "<div class='chat-bubble chat-assistant'>Searching on database "
            "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
            "</div>",
            unsafe_allow_html=True,
        )
        allow_tables = bool(re.search(r"\b(table|tab\.|figure|fig\.)\b", user_input_en.lower()))
        seed_idxs = _strict_retrieve(user_input_en, ds_texts_local or [], ds_index_local, q_emb,
                                     topn_init=40, out_top=6, allow_tables=allow_tables)
        context = _strict_context(ds_texts_local or [], seed_idxs, user_input_en, max_chars=1200,
                                  allow_tables=allow_tables)
        if not context:
            top_chunks = [c for c, _ in (ds_pairs or [])]
            context = _build_context_fallback(top_chunks, user_input_en, max_chars=1200)
        answer_text, sources = generate_zephyr_answer(context, user_input_en, st.session_state.get("chat_history", []), force_web=False)

    elif route == "web":
        placeholder.markdown(
            "<div class='chat-bubble chat-assistant'>Searching on internet "
            "<span class='typing'><span class='dot'>.</span><span class='dot'>.</span><span class='dot'>.</span></span>"
            "</div>",
            unsafe_allow_html=True,
        )
        answer_text, sources = generate_zephyr_answer("", user_input_en, st.session_state.get("chat_history", []), force_web=True)

    else:
        answer_text, sources = generate_zephyr_answer("", user_input_en, st.session_state.get("chat_history", []), force_web=False)

    # Post-process & dil dönüşü
    answer_text = _post_clean_answer(answer_text or "")
    answer_text = _compact_summarize(answer_text, q=user_input_en)
    answer_text = _limit_sentences(answer_text, q=user_input_en)
    if language_selection == "Turkish" and TRANSLATION_AVAILABLE:
        answer_text = translate_from_en(answer_text, "tr")

    uid = str(int(time.time() * 1000))
    chips = render_sources_chips(sources or [], uid)

    bubble_html = f"<div class='chat-bubble chat-assistant'>{answer_text}"
    if chips:
        bubble_html += f"<div class='src-inline'>{chips}</div>"
    bubble_html += "</div>"

    placeholder.markdown(bubble_html, unsafe_allow_html=True)
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer_text,
        "sources": sources or [],
        "uid": uid,
    })
