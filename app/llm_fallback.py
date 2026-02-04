from llama_index.llms.gemini import Gemini
from llama_index.llms.ollama import Ollama
from app.settings import GOOGLE_API_KEY

def get_llm():
    # 1️⃣ Gemini (cloud, free tier)
    if GOOGLE_API_KEY:
        try:
            print("✅ Using Gemini LLM")
            return Gemini(
                model="models/gemini-1.5-flash",
                api_key=GOOGLE_API_KEY,
                temperature=0.2
            )
        except Exception as e:
            print("⚠️ Gemini failed:", e)

    # 2️⃣ Ollama (local, CPU, unlimited)
    print("✅ Using Ollama LLM")
    return Ollama(
        model="llama3:8b",
        temperature=0.2
    )

