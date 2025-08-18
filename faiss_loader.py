# -*- coding: utf-8 -*-
"""
Created on Mon Aug 18 23:50:16 2025

@author: Orhan
"""

from __future__ import annotations
import os
import logging
import tempfile
import requests

try:
    import streamlit as st
except Exception:  # Streamlit yoksa cache'siz çalış
    st = None

try:
    import faiss
except Exception as e:
    raise RuntimeError(
        "faiss kütüphanesi gerekli. requirements.txt içinde faiss-cpu olduğundan "
        "emin olun. Orijinal hata: %r" % e
    )

log = logging.getLogger("faiss_loader")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# Varsayılan Drive dosya ID (istersen .env/Secrets ile override edebilirsin)
DEFAULT_DRIVE_FILE_ID = os.environ.get(
    "FAISS_DRIVE_FILE_ID",
    "1PcSKtFPB0NxTRWQkZETpkRwkvLoIeA_O"  # <- senin verdiğin ID (değiştirebilirsin)
)

def _drive_direct_url(file_id: str) -> str:
    return f"https://drive.google.com/uc?export=download&id={file_id}"

def _download_with_requests(url: str, dest_path: str) -> None:
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    log.info("Downloading FAISS index from: %s", url)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest_path + ".part"
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):  # 1 MB
                if chunk:
                    f.write(chunk)
        os.replace(tmp, dest_path)  # atomic move
    log.info("Saved FAISS index to: %s", dest_path)

def _read_index(path: str):
    idx = faiss.read_index(path)
    try:
        ntotal = int(getattr(idx, "ntotal", 0))
    except Exception:
        ntotal = 0
    log.info("FAISS index loaded. ntotal=%s", ntotal)
    return idx

# ---- Cache decorator (Streamlit varsa) ----
def _cache_resource(func):
    if st is None:
        return func
    return st.cache_resource(show_spinner="Downloading & loading FAISS index…")(func)

@_cache_resource
def load_faiss_index(
    save_dir: str = "rag_index",
    filename: str = "index.faiss",
    file_id: str | None = None,
    force_download: bool = False,
):
    """
    FAISS index'i rag_index/ içine indirip yükler.
    - save_dir/filename path'ine bakar; yoksa Drive'dan indirir.
    - file_id None ise DEFAULT_DRIVE_FILE_ID kullanılır.
    - force_download True ise var olanı da tekrar indirir.
    Dönüş: faiss.Index nesnesi
    """
    file_id = file_id or DEFAULT_DRIVE_FILE_ID
    path = os.path.join(save_dir, filename)
    os.makedirs(save_dir, exist_ok=True)

    if force_download or not os.path.exists(path) or os.path.getsize(path) == 0:
        url = _drive_direct_url(file_id)
        _download_with_requests(url, path)

    # Güvenlik: dosya gerçekten var mı?
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        raise FileNotFoundError(f"FAISS index not found or empty at: {path}")

    return _read_index(path)

def ensure_faiss_index(
    save_dir: str = "rag_index",
    filename: str = "index.faiss",
    file_id: str | None = None,
    force_download: bool = False,
) -> str:
    """
    Sadece indirir/varlığını garanti eder, yüklemez. Dönüş: path (str)
    """
    file_id = file_id or DEFAULT_DRIVE_FILE_ID
    path = os.path.join(save_dir, filename)
    if force_download or not os.path.exists(path) or os.path.getsize(path) == 0:
        _download_with_requests(_drive_direct_url(file_id), path)
    return path

def get_index_status(
    save_dir: str = "rag_index",
    filename: str = "index.faiss",
) -> dict:
    """
    Hızlı durum bilgisi: path, exists, size, ntotal (yüklenirse).
    """
    path = os.path.join(save_dir, filename)
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = {"path": path, "exists": exists, "size_bytes": size, "ntotal": None}
    if exists and size > 0:
        try:
            idx = faiss.read_index(path)
            status["ntotal"] = int(getattr(idx, "ntotal", 0))
        except Exception as e:
            status["ntotal"] = f"read_error: {e}"
    return status

def clear_faiss_cache():
    """Streamlit cache'i temizlemek için yardımcı."""
    if st is not None and hasattr(load_faiss_index, "clear"):
        load_faiss_index.clear()
        log.info("Streamlit resource cache cleared for load_faiss_index().")