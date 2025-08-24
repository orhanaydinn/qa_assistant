# -*- coding: utf-8 -*-
"""
Created on Sat Aug 23 16:08:14 2025

@author: Orhan
"""

import html as htmlmod
import streamlit as st

HERO_HTML = """
<style>
.hero{margin-top:12px;margin-bottom:8px;text-align:center}
.hero h1{display:flex;align-items:center;gap:12px;white-space:nowrap;font-size:clamp(25px,4vw,44px);line-height:1.15;margin:0 0 4px 0;font-weight:300;letter-spacing:-.02em}
.hero h1 .chip{display:inline-flex;align-items:center;padding:0;background:none;border-radius:0;font-weight:600;white-space:nowrap}
.rotate{position:relative;display:inline-block;vertical-align:middle;width:22ch;height:1.3em;overflow:hidden}
.rotate>span{position:absolute;left:0;top:0;opacity:0;transform:translateX(-40px);animation:wordCycle 30s ease-in-out infinite both;white-space:nowrap;color:#3b82f6}
.rotate>span:nth-child(1){animation-delay:0s}.rotate>span:nth-child(2){animation-delay:6s}.rotate>span:nth-child(3){animation-delay:12s}.rotate>span:nth-child(4){animation-delay:18s}.rotate>span:nth-child(5){animation-delay:24s}
@keyframes wordCycle{0%{opacity:0;transform:translateX(-40px)}8%{opacity:1;transform:translateX(0)}16%{opacity:1;transform:translateX(0)}20%{opacity:0;transform:translateX(-40px)}100%{opacity:0;transform:translateX(-40px)}}
.hero .muted{color:#6b7280;font-size:clamp(14px,2vw,16px);margin-top:6px}
@media (prefers-color-scheme: dark){.hero .muted{color:#9ca3af}}
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
"""

CHAT_CSS = """
<style>
.chat-bubble{padding:10px 14px;border-radius:12px;margin:4px 0 6px;word-wrap:break-word;overflow-wrap:anywhere;white-space:pre-wrap;line-height:1.45}
.chat-user{background:#ccecff;color:#000;margin-left:auto;text-align:left;width:fit-content;max-width:60%}
.chat-assistant{background:#f0f0f0;color:#000;margin-right:auto;text-align:left;width:100%}
.typing .dot{display:inline-block;opacity:0;animation:blink 1.2s infinite}
.typing .dot:nth-child(2){animation-delay:.2s}.typing .dot:nth-child(3){animation-delay:.4s}
@keyframes blink{0%{opacity:0}20%{opacity:1}60%{opacity:1}100%{opacity:0}}
@media (max-width:640px){.chat-user{max-width:90%}}
.chat-assistant .src-inline{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px;padding:0;line-height:1.2}
.src-chip{font-size:12px;background:#f3f4f6;color:#374151;padding:2px 8px;border-radius:9999px;text-decoration:none;white-space:nowrap;display:inline-flex;align-items:center;gap:6px;cursor:pointer}
.src-chip:hover{background:#e5e7eb;text-decoration:underline}
.src-chip.link::before{content:"🌐";font-size:13px;line-height:1}
.extra-sources{display:none;margin:2px 0 0;padding:0;gap:6px;flex-wrap:wrap}
.extra-sources .src-chip::before{content:"🔗"}
img.gen{max-width:100%;border-radius:12px;margin-top:8px;border:1px solid #e5e7eb}
</style>
"""

TOGGLE_JS = """
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
"""

FOOTER_HTML = """
<style>
.input-container{text-align:center;margin-top:-6px}
.input-footer{text-align:center;font-size:12px;color:#6b7280;margin-top:0}
.input-footer a{color:#6b7280;text-decoration:none}
.input-footer a:hover{text-decoration:underline}
@media (prefers-color-scheme: dark){.input-footer{color:#9ca3af}.input-footer a{color:#9ca3af}}
</style>
<div class="input-container">
  <div class="input-footer">
    This project developed by <a href="https://www.linkedin.com/in/orhan-aydin/" target="_blank"><b>Orhan Aydin</b></a>.
  </div>
</div>
"""

def render_sources_chips(sources, uid: str) -> str:
    if not sources:
        return ""
    main_src = sources[0]
    url = main_src.get("url")
    domain = (main_src.get("source") or main_src.get("title") or "Source")
    extra = len(sources) - 1

    chips_inner = f"<a href='{url}' target='_blank' class='src-chip link'>{domain}</a>"
    extra_html = ""
    if extra > 0:
        chips_inner += (f"<span class='src-chip toggle-chip' data-target='extra-{uid}' data-count='{extra}'>+{extra}</span>")
        extra_html = f"<div id='extra-{uid}' class='extra-sources' style='display:none;'>"
        for s in sources[1:]:
            su = s.get("url")
            sd = (s.get("source") or s.get("title") or "Source")
            extra_html += f"<a href='{su}' target='_blank' class='src-chip link'>{sd}</a>"
        extra_html += "</div>"
    return chips_inner + extra_html

def render_history():
    for i, msg in enumerate(st.session_state.chat_history):
        if msg["role"] == "user":
            st.markdown(
                f"<div class='chat-bubble chat-user'>{htmlmod.escape(msg['content'])}</div>",
                unsafe_allow_html=True,
            )
        else:
            uid = msg.get("uid") or f"h{i}"
            if msg.get("image_b64"):
                caption = msg.get("content", "Image")
                try:
                    _label, _cap = caption.split(":", 1)
                    _cap = _cap.strip()
                except ValueError:
                    _label, _cap = "Image", caption
                img_html = (
                    f"<div class='chat-bubble chat-assistant'>"
                    f"<div><strong>{htmlmod.escape(_label)}:</strong> {htmlmod.escape(_cap)}</div>"
                    f"<img class='gen' src='data:{msg.get('mime','image/png')};base64,{msg['image_b64']}' alt='generated image'/>"
                    f"</div>"
                )
                st.markdown(img_html, unsafe_allow_html=True)
                continue

            content_html = msg.get("content", "")
            sources_hist = msg.get("sources") or []
            chips_inner, extra_html = "", ""
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
