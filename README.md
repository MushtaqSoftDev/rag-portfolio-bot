🤖 RAG Portfolio Backend

A high-performance Retrieval-Augmented Generation (RAG) backend designed to power a 3D portfolio. This system allows users to interact with Mushtaq professional data via a smart chat interface, ensuring responses are grounded in real project data.

🏗️ Architecture

    Framework: FastAPI (Python)
    RAG Engine: LlamaIndex
    Data Source: Local markdown/text files in /data
    Deployment: Render.com (Free Tier)
    Persistence: Local disk storage in /storage (persisted index)
    
    
🚀 Key Features
1. Multi-LLM Fallback System

The system is built to be resilient. It automatically switches between models based on availability and API limits:

    Primary: Groq Llama 3 (High reasoning, free).

    Secondary: Gemini Pro (Ultra-fast inference fallback & free tier).

    Emergency: TinyLlama (Local CPU/Hugging face model) — ensures the bot never goes offline, even if all API quotas are exhausted.

2. Zero-Latency "Warm-Up" Logic

To bypass Render's 15-minute "Free Tier Sleep," this backend includes a lightweight health-check route.

    Endpoint: GET /

    Function: Returns a simple {"status": "ok"} without triggering LLM logic.

    Optimization: Keeps the server awake 24/7 via external pings.

3. Contextual Guardrails

The engine is configured with a custom system prompt that prevents "hallucinations" or off-topic discussions. It is strictly instructed to only answer questions related to Mushtaq portfolio, projects, and technical skills.


📁 Project Structure
├── app/
│   ├── api.py          # API Route definitions (/chat)
│   ├── rag_engine.py   # LlamaIndex logic (indexing & querying)
│   ├── settings.py     # Global configuration & LLM Fallback logic
│
|__ data                # Portfolio source documents (MD)
|__ scripts             # Make ingestion & script to test locally
├── storage/            # Persisted vector index files
├── main.py             # FastAPI entry point & Health Check
├── requirements.txt    # Project dependencies
└── Dockerfile          # Container configuration for Render


🛠️ Installation & Setup

    1. Clone the Repository
    https://github.com/MushtaqSoftDev/rag-portfolio-bot
    cd rag-portfolio-bot

    2. Environment Setup
    Create a .env file in the root directory:
    GOOGLE_API_KEY=your_gemini_api_key
    GROQ_API_KEY=your_groq_api_key

    3. Install Dependencies
    pip install -r requirements.txt

    4. Run Locally
    python main.py