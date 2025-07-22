# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:17:47 2025

@author: Orhan
"""

import fitz  # PyMuPDF

def extract_text_chunks(file, chunk_size=150):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    words = text.split()
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks
