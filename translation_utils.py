from langdetect import detect
from deep_translator import GoogleTranslator
import re

SUPPORTED_LANGUAGES = {
    "en": "english",
    "tr": "turkish",
    "es": "spanish",
    "fr": "french",
    "de": "german",
    "hi": "hindi",
    "zh": "chinese (simplified)"
}

language_aliases = {
    "en": ["english", "ingilizce", "İngilizce", "ENGLISH", "Englisch", "INGLIZCE", "Ingilizce", "İnglish", "eng", "anglès", "angielski", "ingles", "inglese", "englanti", "английский", "en", "eng. lang", "e. lang", "engl", "en-lang", "english lang", "inglis", "en pls", "eng response", "english version", "eng only", "output in english", "answer english", "in english please", "give me english", "please use english", "use english lang", "switch to english", "eng language", "in eng", "englsh", "engliş", "englîsh", "engliç", "anglais", "eng-lisan", "eng dil", "en dilinde", "engce", "engçe", "angl", "english text", "text in english", "yaz ingilizce", "ing"],
    "tr": ["turkish", "türkçe", "Türkçe", "TURKCE", "Turkce", "turkçe", "turkısh", "turquía", "tr", "турецкий", "turc", "tuerkisch", "turki", "turque", "turca", "tr dil", "tr dili", "tr-lang", "turk lang", "turkish lang", "turkish language", "turk language", "türk dili", "türk dilinde", "türk cevap", "cevap türkçe", "türk yaz", "türkçe yaz", "turkce yaz", "tur version", "turk versiyon", "yaz türkçe", "tur. lang", "t dil", "türkce", "türkçee", "türkc", "tr only", "output in turkish", "turk lisan", "tur. version", "turkish version", "tr text", "turkish response", "türkçe cevap ver", "tr response", "tur. dil", "türkçe anlat", "türkçe göster", "türkçe lütfen"],
    "es": ["spanish", "español", "Español", "espanol", "espanhol", "ESPANOL", "Ispanyolca", "ispanyolca", "espaniol", "española", "espanjol", "espanyol", "es", "es-lang", "espanolce", "espanol dil", "espanolca", "output in spanish", "respuesta en español", "espanol versiyon", "text in spanish", "spanish response", "spanish only", "spanish please", "responde en español", "espanol yaz", "cevap espanol", "spanish yaz", "es version", "espanol version", "espanol dilinde", "espanol dil", "es. lang", "esp dil", "esp lisan", "respuesta espanola", "en español por favor", "quiero respuesta en español", "usa español", "espanol kullan", "dame la respuesta en español", "respuesta solo en español", "muestra la respuesta en español", "salida en español"],
    "fr": ["french", "français", "Français", "francais", "francese", "franse", "FRANCAIS", "française", "fransızca", "Fransızca", "fr", "fr-lang", "fr dil", "fr dili", "french lang", "langue française", "réponds en français", "fr response", "texte en français", "version française", "en français s'il vous plaît", "utiliser le français", "afficher en français", "french only", "output in french", "fransızca yaz", "yaz fransızca", "fr dilinde", "cevap fransızca", "french cevap", "french yaz", "french metin", "fr metni", "fr lisan", "fr langue", "dil français", "réponse en français", "donne la réponse en français", "réponse française", "affiche la réponse en français", "en français svp"],
    "de": ["german", "deutsch", "Deutsch", "almanca", "Almanca", "alemán", "alemao", "allemand", "tedesco", "deutsche", "de", "niemcy", "de-lang", "de dil", "de dili", "deutsch lang", "langue allemande", "antwort auf deutsch", "de response", "text auf deutsch", "german version", "auf deutsch bitte", "verwende deutsch", "zeige auf deutsch", "deutsch only", "output in german", "almanca yaz", "yaz almanca", "de dilinde", "cevap almanca", "german cevap", "german yaz", "german metin", "de metni", "de lisan", "de langue", "sprache deutsch", "antwort auf deutsch geben", "gib die antwort auf deutsch", "deutsche antwort", "zeige die antwort auf deutsch", "auf deutsch svp"],
    "hi": ["hindi", "हिंदी", "Hindi", "hindî", "हिन्दी", "HINDI", "hintçe", "Hintçe", "हिंदी में", "हिंदी भाषा", "हिंदी उत्तर", "हिंदी में जवाब", "hindustani", "हिंदुस्तानी", "hi", "hi-lang", "हिंदी लिपि", "output in hindi", "text in hindi", "उत्तर हिंदी में", "उत्तर दें हिंदी में", "हिंदी में उत्तर चाहिए", "हिंदी लिखिए", "हिंदी में दिखाएँ", "जवाब हिंदी में होना चाहिए", "हिंदी उत्तर दीजिए", "हिंदी जवाब दो", "हिंदी उपयोग करें", "हिंदी लिपि में", "हिंदी भाषा में", "हिंदी में उत्तर दीजिए", "हिंदी में जवाब दें", "उत्तर हिंदी में होना चाहिए", "कृपया हिंदी में जवाब दें", "कृपया हिंदी में उत्तर दें", "हिंदी में बताइए", "हिंदी में दिखाइए", "हिंदी में उत्तर दर्ज करें", "हिंदी उत्तर चाहिए", "हिंदी में उत्तर दीजिए", "हिंदी में लिखिए", "कृपया हिंदी का प्रयोग करें", "उत्तर दीजिए हिंदी में", "हिंदी में उत्तर दिखाइए"],
    "zh": ["chinese", "Chinese", "中文", "汉语", "汉文", "Çince", "çince", "mandarin", "putonghua", "zhongwen", "chinês", "chinees", "zh", "汉字", "中国话", "汉话", "中國語", "中文回答", "zh-lang", "ch lang", "output in chinese", "text in chinese", "用中文回答", "请用中文回答", "回答用中文", "請用中文回答", "用汉语回答", "用汉语写", "请使用中文", "请用简体中文回答", "用普通话回答", "请将答案写成中文", "显示中文答案", "写中文", "写在中文里", "中文版本", "请输出中文", "用中国语言", "使用中文语言", "请用中国话", "中文答复", "请用中文答复", "翻译成中文", "用简体中文"]
}

phrase_templates = [
    "in {}", "answer in {}", "please answer in {}", "respond in {}", "reply in {}",
    "write in {}", "use {}", "switch to {}", "{} please", "output in {}",
    "show answer in {}", "respond using {}", "give me the answer in {}",
    "could you answer in {}", "{} answer", "make it {}", "convert to {}",
    "display answer in {}", "translate to {}", "{} language", "talk in {}",
    "write response in {}", "say it in {}", "respond with {}",
    "please do it in {}", "{} only", "keep it in {}", "explain in {}",
    "respond entirely in {}", "output must be in {}", "{} response only",
    "everything in {}", "switch language to {}", "communicate in {}",
    "generate in {}", "generate response in {}", "use {} language",
    "cevabı {} ver", "yanıtı {} yaz", "{} yaz", "{} olsun", "lütfen {} yaz",
    "{} olarak yanıtla", "cevap {} olabilir mi", "çıktı {} olsun",
    "{} dönüş yap", "yanıt {} dilinde olsun", "metni {} yaz", "{} anlat",
    "yanıtı {} olarak belirt", "{} versiyonunu göster", "cevabı {}ye çevir",
    "{} diliyle yaz", "{} dilini kullan", "lütfen {} dilinde cevapla",
    "yanıt {} dilinde ver", "{} dilinde cevap yaz", "metni {} dilinde göster",
    "{} formatında yanıtla", "{} ile açıkla", "{} diliyle açıkla",
    "कृपया {} में उत्तर दें", "请用{}回答", "réponds en {}", "responde en {}",
    "antwort auf {}", "trả lời bằng {}", "답변을 {}로 해주세요", "{} bahasa gunakan",
    "{} में उत्तर दें", "답을 {}로 작성하세요", "{} में जवाब दें", "{} में लिखें",
    "{} का उपयोग करें", "{} में बोलें", "{} زبان میں جواب دیں"
]

def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

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

def extract_target_language_instruction(text):
    text_lower = text.casefold()
    for lang_code, aliases in language_aliases.items():
        for alias in aliases:
            for template in phrase_templates:
                phrase = template.format(alias).casefold()
                if phrase in text_lower:
                    return lang_code
    return None

def extract_translation_instruction(text):
    import re
    text_cf = text.casefold()
    target_lang = None
    content = None

    # Desteklenen tetikleyici ifadeler
    triggers = [
        "bunu", "bu bilgiyi", "cevabı", "şunu", "yukarıdaki", "önceki mesajı",
        "bu açıklamayı", "bu metni", "bilgiyi", "içeriği"
    ]

    for lang_code, aliases in language_aliases.items():
        for alias in aliases:
            for trigger in triggers:
                patterns = [
                    f"{trigger} {alias}e çevir",
                    f"{trigger} {alias}ye çevir",
                    f"{trigger} {alias} diline çevir",
                    f"{trigger} {alias} çevirebilir misin",
                    f"{trigger} {alias} yap",
                    f"{trigger} {alias} olsun",
                    f"{trigger} {alias} versiyonunu göster",
                    f"{trigger} {alias} diliyle yaz",
                    f"{trigger} {alias} dilinde yaz",
                    f"{trigger} {alias} şeklinde göster",
                    f"{trigger} {alias} dilinde lütfen",
                    f"{trigger} {alias} olarak çevir"
                ]
                for p in patterns:
                    if p in text_cf:
                        target_lang = lang_code
                        break
            if target_lang:
                break
        if target_lang:
            break

    # Çevirilecek içerik ayıklama
    if target_lang:
        content_match = re.split(
            r"(bunu .*? çevir|translate this to .*?|convert this to .*?|please translate to .*?|please convert to .*?)",
            text, flags=re.IGNORECASE
        )
        if content_match:
            content = content_match[0].strip()
        return target_lang, content

    return None, None

