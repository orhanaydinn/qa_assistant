# -*- coding: utf-8 -*-
"""
Created on Tue Jul 22 16:17:47 2025

@author: Orhan
"""

import pdfplumber

def extract_text_chunks(pdf_path):
    text_chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                text_chunks.append(text)
    return text_chunks