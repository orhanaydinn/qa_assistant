import os
import requests
from huggingface_hub import InferenceClient

client = InferenceClient(
    model="HuggingFaceH4/zephyr-7b-alpha",
    token=os.getenv("HF_API_TOKEN")
)

SERPER_API_KEY = os.getenv("SERPER_API_KEY")

def get_web_summary_serper(query):
    try:
        headers = {
            "X-API-KEY": SERPER_API_KEY,
            "Content-Type": "application/json"
        }
        data = {"q": query}
        res = requests.post("https://google.serper.dev/search", headers=headers, json=data, timeout=8)
        if res.status_code != 200:
            return ""
        json_data = res.json()

        snippets = []
        if "answerBox" in json_data and "answer" in json_data["answerBox"]:
            snippets.append(json_data["answerBox"]["answer"])
        for item in json_data.get("organic", [])[:2]:
            if "snippet" in item:
                snippets.append(item["snippet"])
        return " ".join(snippets).strip()
    except Exception as e:
        return ""

def needs_web_context(question):
    keywords = [
        "2024", "2023", "today", "now", "current", "latest", "this year", "recent",
        "weather", "election", "result", "news", "price", "score", "exchange rate",
        "who won", "when is", "what time", "live", "president", "prime minister"
    ]
    return any(kw in question.lower() for kw in keywords)

def is_response_broken(text):
    weirdness_score = sum(1 for c in text if ord(c) > 1000 or c in "¼½¾™®©•§µ")
    has_fake_q = any(kw in text.lower() for kw in [
        "user:", "question:", "q:", "generate according to", "recent exchange", "note:"
    ])
    return (
        len(text) > 2000 or
        weirdness_score > 5 or
        has_fake_q or
        any(bad in text.lower() for bad in [
            "retrieved from", "last revised", "wikipedia.org",
            "porn", "xxx", "tube8", "custom essay"
        ])
    )

def clean_bad_patterns(text):
    bad_tokens = [
        "user:", "question:", "q:", "generate according to", "recent exchange", "note:"
    ]
    for token in bad_tokens:
        if token in text.lower():
            return text.split(token)[0].strip()
    return text

def generate_zephyr_answer(context, question, history=None):
    used_web = False
    status_message = "Generating answer..."

    if needs_web_context(question):
        web_info = get_web_summary_serper(question)
        if web_info.strip():
            context = f"[WEB RESULT]\n{web_info.strip()}\n\n{context}"
            used_web = True
            status_message = "Searching the internet..."
        else:
            return "⚠️ No reliable up-to-date information was found online.", "Searching the internet..."

    history_prompt = ""
    if history:
        last_n = history[-1:] if len(history) >= 1 else history
        for turn in last_n:
            history_prompt += f"{turn['user']}\n{turn['bot']}\n"

    DETAIL_KEYWORDS = [
        "example", "give an example", "show an example", "for instance",
        "explain more", "explain further", "more detail", "detailed",
        "can you elaborate", "elaborate", "expand", "walk me through",
        "how does it work", "how does this work", "what does that mean",
        "what do you mean", "clarify", "can you clarify", "describe",
        "in depth", "step by step", "demonstrate", "could you show",
        "show me how", "could you explain that better", "break it down",
        "break down", "go deeper", "give more information", "deep dive",
        "drill down", "case study", "use case", "sample use", "walkthrough",
        "further explanation", "what's an example", "what's a use case", "specific scenario"
    ]

    wants_detail = any(keyword in question.lower() for keyword in DETAIL_KEYWORDS)
    last_user_turn = history[-1]["user"] if history else ""

    if question.strip().lower() in ["give an example", "can you give an example?"] and last_user_turn:
        effective_question = f"{last_user_turn} {question}"
    elif wants_detail and len(question.split()) <= 8 and last_user_turn:
        effective_question = f"{last_user_turn} {question}"
    else:
        effective_question = question

    style_instruction = (
        "If the user asks for more detail or an example, respond with 3–5 concise sentences. "
        "Include one focused example only. Do not go off-topic or provide multiple examples."
        if wants_detail
        else "Answer clearly and concisely in 2–3 sentences. Do not add extra explanation unless asked."
    )

    prompt = f"""
You are a helpful and knowledgeable AI assistant.

{style_instruction}

Use the context and short chat history below to answer the user's current question only.

Do not invent new questions.
Do not continue the conversation unless asked.
Do not include phrases like 'User:', 'Question:', 'Recent exchange:', 'Note:', or similar formatting in your answer.

Context:
{context}

Recent exchange:
{history_prompt}

User question:
{effective_question}
""".strip()

    try:
        response = client.chat.completions.create(
            model="HuggingFaceH4/zephyr-7b-alpha",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful and concise assistant. Do not ask your own questions. Do not continue the conversation. Only respond directly to the user input."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=250
        )

        if not response or not response.choices or not response.choices[0].message:
            return "⚠️ The assistant could not generate a valid response. Please try again.", status_message

        answer = response.choices[0].message.content.strip()

        if answer.lower().startswith("assistant:"):
            answer = answer[len("assistant:"):].strip()

        answer = clean_bad_patterns(answer)

#        if is_response_broken(answer):
#            return "⚠️ The assistant generated an invalid or off-topic response. Please try rephrasing your question.", status_message

        return answer, status_message

    except Exception as e:
        return f"⚠️ Error during API call: {e}", status_message
