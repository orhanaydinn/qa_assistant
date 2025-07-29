# Güncellenmiş translation_utils.py içeriğini oluşturalım
import csv
import re
import json
from langdetect import detect
from deep_translator import GoogleTranslator

# ------------------------
# 📥 Türkçe kelime seti yükleniyor
# ------------------------

try:
    with open("C:/Users/Orhan/Desktop/Software Project/Artificial Intelligence/Projects/pdf_qa_project_v4_TEST/Dataset/tdk_word_data_all.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        turkish_word_set = set()
        for row in reader:
            if row:  # Boş satır kontrolü
                word = row[0].strip()
                if word:
                    turkish_word_set.add(word)
except Exception as e:
    print("TDK kelime dosyası yüklenemedi:", e)
    turkish_word_set = set()
# ------------------------
# 🔍 Türkçe kelime sinyali
# ------------------------
def score_turkish_signal(text):
    words = re.findall(r"\\b\\w+\\b", text.lower())
    return sum(1 for word in words if word in turkish_word_set)

# ------------------------
# 🧠 Güvenli dil tespiti (sadece 'tr' ve 'en')
# ------------------------
def smart_detect_language(text, threshold=2):
    try:
        base_lang = detect(text)
    except:
        base_lang = "en"

    if base_lang not in ["tr", "en"]:
        print(f"[DEBUG] Langdetect detected unsupported language '{base_lang}', defaulting to 'en'")
        base_lang = "en"

    score = score_turkish_signal(text)
    print(f"[DEBUG] Langdetect: {base_lang}, Turkish word count: {score}")

    if base_lang == "en" and score >= threshold:
        print("[DEBUG] Forcing language to 'tr' based on Turkish keyword score.")
        return "tr"

    return base_lang

# ------------------------
# 🌍 Çeviri fonksiyonları
# ------------------------
SUPPORTED_LANGUAGES = {
    "en": "english",
    "tr": "turkish"
}

def translate_to_en(text, source_lang=None):
    try:
        source = SUPPORTED_LANGUAGES.get(source_lang, "auto")
        return GoogleTranslator(source=source, target="english").translate(text)
    except:
        return text

def translate_from_en(text, target_lang="en"):
    try:
        target = SUPPORTED_LANGUAGES.get(target_lang, "english")
        return GoogleTranslator(source="english", target=target).translate(text)
    except:
        return text

# ------------------------
# 🔁 Dil yönlendirme alias + templates
# ------------------------
language_aliases = {
    "en": [
        "english", "eng", "en", "in english", "english please", "answer in english",
        "english output", "english version", "eng answer", "eng yaz", "output in english"
    ],
    "tr": [
        "türkçe", "turkce", "tr", "in turkish", "cevap türkçe", "türkçe olsun",
        "cevabı türkçe ver", "output in turkish", "write in turkish", "tr yaz"
    ]
}

phrase_templates = [
    "in {}", "answer in {}", "please answer in {}", "{} please", "{} only",
    "output in {}", "output must be in {}", "respond in {}", "{} olsun",
    "cevabı {} ver", "{} yanıtla", "{} olarak cevapla"
]

# ------------------------
# 🎯 Hedef yanıt dili çıkarımı
# ------------------------
def extract_target_language_instruction(text):
    text_lower = text.casefold()

    for lang_code, aliases in language_aliases.items():
        for alias in aliases:
            if alias.casefold() in text_lower:
                return lang_code

    for lang_code, aliases in language_aliases.items():
        for alias in aliases:
            for template in phrase_templates:
                phrase = template.format(alias).casefold()
                if phrase in text_lower:
                    return lang_code

    return None


