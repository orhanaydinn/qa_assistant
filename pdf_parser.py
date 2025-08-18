# -*- coding: utf-8 -*-
"""
PDF -> metin çıkarma ve yapısal chunk'lama (tablo/başlık/listeleri korur)
- Tablo başlığı "Table 1 ..." tespit edilir ve tablo bloğu şu şekilde etiketlenir:
  [START_TABLE id=1 title="..."]
  ... tablo satırları ...
  [END_TABLE]
- Çıkış: List[str]  (app.py ile uyumlu)
- Her chunk başında sayfa/segment bilgisi: "Page {p} [{i}/{n}]: ..."
"""
from typing import List, Tuple, Union, Optional
import io
import re
import statistics
import fitz  # PyMuPDF

# Hedef uzunluk / overlap
CHUNK_CHAR_LIMIT = 1600
CHUNK_CHAR_OVERLAP = 250

# Heuristik eşikler
HEADING_SIZE_DELTA = 2.0     # medyan punto + delta -> başlık adayı
MIN_HEADING_LEN = 3
MAX_HEADING_LEN = 140

LIST_BULLETS = ("•", "-", "–", "*", "·", "○", "▪", "‣")


def _read_pdf_bytes(pdf_file: Union[io.BytesIO, bytes, str]):
    if isinstance(pdf_file, (bytes, bytearray)):
        return bytes(pdf_file)
    if hasattr(pdf_file, "read"):
        data = pdf_file.read()
        try:
            pdf_file.seek(0)
        except Exception:
            pass
        return data
    if isinstance(pdf_file, str):
        return None  # path için fitz.open(filename=...) kullanacağız
    raise TypeError("Unsupported pdf_file type. Use path, bytes, or a file-like with .read().")


def _normalize_text(s: str) -> str:
    if not s:
        return ""
    # tire ile satır sonu birleştirme
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)
    # CRLF -> LF -> boşluk
    s = s.replace("\r", "\n")
    # fazla boşluk temizliği
    s = re.sub(r"[ \t]+", " ", s)
    # paragraf boşlukları kalsın
    s = re.sub(r"\n\s*\n+", "\n\n", s)
    # tek satır sonlarını boşluğa indir
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def _is_list_line(line: str) -> bool:
    if not line:
        return False
    line = line.strip()
    if any(line.startswith(b) for b in LIST_BULLETS):
        return True
    # "1.", "1)", "a)", "i." gibi
    if re.match(r"^(\d+[\.\)])\s+", line):
        return True
    if re.match(r"^[a-zA-Z][\)\.]\s+", line):
        return True
    return False


def _is_table_like(line: str) -> bool:
    """
    Basit tablo sezgisi:
    - satırda 2+ boşlukla hizalanmış sütunlar (çoklu spacing)
    - '|' ya da ';' ile ayrılmış kolonlar
    - ardışık kısa token'lar (çok sayıda kısa kelime)
    """
    if not line:
        return False
    if "|" in line or ";" in line:
        return True
    # 2+ space ile sütun ayırımı
    if re.search(r"\s{2,}\S+\s{2,}", line):
        return True
    # bir satırda 8+ kısa token varsa tablo olma ihtimali
    toks = line.split()
    short_tokens = sum(1 for t in toks if len(t) <= 4)
    if len(toks) >= 8 and short_tokens >= 5:
        return True
    return False


_TABLE_HDR_RE = re.compile(r"^(Table|Tab\.)\s*(\d+)\s*[:\-–]?\s*(.*)$", re.IGNORECASE)

def _parse_table_heading(text: str) -> Optional[Tuple[int, str]]:
    """
    'Table 1: Title...' gibi başlıktan (numara, başlık) döndürür.
    """
    if not text:
        return None
    line = text.strip()
    m = _TABLE_HDR_RE.match(line)
    if not m:
        return None
    num = int(m.group(2))
    title = (m.group(3) or "").strip()
    # Çok kısa/boş başlık ise boş bırakabiliriz
    return num, title


def _tag_table_block(text: str, heading_text: Optional[str]) -> str:
    """
    Tablo bloğunu [START_TABLE id=.. title=".."] ... [END_TABLE] ile sarar.
    Başlık yoksa id/title yazmadan etiketler.
    """
    tid_title = _parse_table_heading(heading_text or "")
    if tid_title:
        tid, title = tid_title
        if title:
            safe_title = title.replace('"', "'")
            title_attr = f' title="{safe_title}"'
        else:
            title_attr = ""
        return f"[START_TABLE id={tid}{title_attr}]\n{text}\n[END_TABLE]"
    # Bazı dokümanlarda tablo başlığı bloğun ilk satırı olabilir; bir kere daha dene
    first_line = (text.split("\n", 1)[0] if "\n" in text else text).strip()
    tid_title2 = _parse_table_heading(first_line)
    if tid_title2:
        tid, title = tid_title2
        if title:
            safe_title = title.replace('"', "'")
            title_attr = f' title="{safe_title}"'
        else:
            title_attr = ""
        return f"[START_TABLE id={tid}{title_attr}]\n{text}\n[END_TABLE]"
    return f"[START_TABLE]\n{text}\n[END_TABLE]"


def _extract_page_blocks(page) -> List[Tuple[float, str]]:
    """
    Sayfadaki metni 'rawdict' ile alıp span punto değerlerine ulaşır,
    blokları bir araya getirir. Her dönen öğe: (yaklaşık-font-size, metin)
    """
    raw = page.get_text("rawdict")
    blocks = raw.get("blocks", [])
    items: List[Tuple[float, str]] = []

    for b in blocks:
        # sadece text block
        if b.get("type") != 0:
            continue
        block_lines = []
        sizes = []
        for line in b.get("lines", []):
            line_text = ""
            line_sizes = []
            for span in line.get("spans", []):
                t = span.get("text", "")
                if not t:
                    continue
                line_text += t
                sz = float(span.get("size", 0.0) or 0.0)
                if sz > 0:
                    line_sizes.append(sz)
            if line_text.strip():
                block_lines.append(line_text.strip())
                if line_sizes:
                    sizes.extend(line_sizes)
        if not block_lines:
            continue
        block_text = "\n".join(block_lines)
        median_size = statistics.median(sizes) if sizes else 0.0
        items.append((median_size, block_text))
    return items


def _is_heading(text: str, size: float, page_sizes_median: float) -> bool:
    if not text:
        return False
    t = text.strip()
    if len(t) < MIN_HEADING_LEN or len(t) > MAX_HEADING_LEN:
        return False
    # Büyük punto + kısa satır -> başlık adayı
    if size >= (page_sizes_median + HEADING_SIZE_DELTA):
        return True
    # "Table 1", "Figure 2" gibi açıklayıcı başlıklar
    if _TABLE_HDR_RE.match(t):
        return True
    # Tamamen büyük harf ve kısa ise (SECTION HEADERS gibi)
    if t.isupper() and len(t.split()) <= 10:
        return True
    return False


def _group_structured_blocks(blocks: List[Tuple[float, str]]) -> List[str]:
    """
    Blokları şu etiketlerle birleştirir:
    [H] Başlık
    [LIST] Liste bloğu
    [START_TABLE id=.. title=".."] ... [END_TABLE] Tablo bloğu
    Paragrafları birleştirirken uzun olursa limitte kırar (overlap'lı).
    """
    if not blocks:
        return []

    # sayfa median punto (başlık tespiti için referans)
    page_sizes = [sz for (sz, _t) in blocks if sz > 0]
    page_med = statistics.median(page_sizes) if page_sizes else 0.0

    out_parts: List[str] = []
    cur_para = ""
    last_heading_text: Optional[str] = None  # tablo bloğuna başlık numarasını iliştirmek için

    def flush_para():
        nonlocal cur_para
        cur_para = cur_para.strip()
        if not cur_para:
            return
        if len(cur_para) <= CHUNK_CHAR_LIMIT:
            out_parts.append(cur_para)
        else:
            # overlap'lı böl
            txt = cur_para
            while txt:
                piece = txt[:CHUNK_CHAR_LIMIT]
                out_parts.append(piece)
                if len(txt) <= CHUNK_CHAR_LIMIT:
                    break
                start = CHUNK_CHAR_LIMIT - CHUNK_CHAR_OVERLAP
                if start < 0:
                    start = 0
                txt = txt[start:]
        cur_para = ""

    for sz, raw in blocks:
        text = _normalize_text(raw)
        if not text:
            continue

        # Başlık?
        if _is_heading(text, sz, page_med):
            flush_para()
            out_parts.append(f"[H] {text}")
            last_heading_text = text  # tablo numarası için sakla
            continue

        # Liste bloğu mu?
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        list_ratio = sum(1 for l in lines if _is_list_line(l)) / max(1, len(lines))
        is_list_block = list_ratio >= 0.6 and len(lines) >= 2

        if is_list_block:
            flush_para()
            out_parts.append("[LIST]\n" + "\n".join(lines))
            continue

        # Tablo benzeri?
        if any(_is_table_like(l) for l in lines):
            flush_para()
            out_parts.append(_tag_table_block("\n".join(lines), last_heading_text))
            continue

        # Normal paragraf: mevcut paragraf ile birleştir
        if not cur_para:
            cur_para = text
        else:
            cur_para += "\n\n" + text

        # Çok uzadıysa kır
        if len(cur_para) > (CHUNK_CHAR_LIMIT * 1.4):
            flush_para()

    flush_para()
    return out_parts


def extract_text_chunks(pdf_file) -> List[str]:
    """
    PDF'den metni yapısal olarak çıkar ve küçük chunk'lara böl.
    - pdf_file: path, bytes veya .read() destekleyen obje (Streamlit UploadedFile/BytesIO)
    - dönüş: List[str]
    Her chunk başına "Page {p} [i/n]:" ön eki eklenir.
    """
    chunks: List[str] = []

    pdf_bytes = _read_pdf_bytes(pdf_file)
    if pdf_bytes is None and isinstance(pdf_file, str):
        doc = fitz.open(pdf_file)
    else:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    for pno, page in enumerate(doc, start=1):
        # sayfayı blok+span düzeyinde çek
        blocks = _extract_page_blocks(page)
        if not blocks:
            # fallback: düz metin
            raw = page.get_text()
            text = _normalize_text(raw)
            if not text:
                continue
            parts = _group_structured_blocks([(0.0, text)])
        else:
            parts = _group_structured_blocks(blocks)

        if not parts:
            continue

        n = len(parts)
        for i, part in enumerate(parts, start=1):
            header = f"Page {pno} [{i}/{n}]: "
            chunks.append(header + part.strip())

    return chunks
