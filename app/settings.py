import os
import torch
from dotenv import load_dotenv

from llama_index.core.settings import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

load_dotenv()

# --------------------------------------------------
# Embeddings (LOCAL, FAST, REQUIRED)
# --------------------------------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
USE_GEMINI = os.getenv("USE_GEMINI", "false").lower() == "true"


def get_llm():
    # --------------------------------------------------
    # 1️⃣ GROQ (FASTEST)
    # --------------------------------------------------
    if USE_GROQ:
        try:
            from llama_index.llms.groq import Groq

            print("🚀 Using Groq LLM")
            return Groq(
                model="llama-3.1-8b-instant",
                temperature=0.2,
            )
        except Exception as e:
            print("⚠️ Groq failed, trying Gemini...")
            print(e)

    # --------------------------------------------------
    # 2️⃣ GEMINI (FALLBACK)
    # --------------------------------------------------
    if USE_GEMINI:
        try:
            from llama_index.llms.gemini import Gemini

            print("✨ Using Gemini LLM")
            return Gemini(
                model="models/gemini-pro",
                temperature=0.2,
            )
        except Exception as e:
            print("⚠️ Gemini failed, falling back to TinyLlama...")
            print(e)

    # --------------------------------------------------
    # 3️⃣ LOCAL HF (EMERGENCY)
    # --------------------------------------------------
    print("🧠 Using TinyLlama (local CPU fallback)")

    return HuggingFaceLLM(
        model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.2},
        model_kwargs={
            "dtype": torch.float32,
            "low_cpu_mem_usage": True,
        },
    )


Settings.llm = get_llm()

# --------------------------------------------------
# Performance tuning
# --------------------------------------------------
Settings.context_window = 2048
