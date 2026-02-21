🤖 RAG Portfolio Backend

A high-performance Retrieval-Augmented Generation (RAG) backend designed to power a 3D portfolio. This system allows users to interact with Mushtaq's professional data via a smart chat interface, ensuring responses are grounded in real project data.

## Agentic RAG (not Linear RAG)

This project uses **Agentic RAG**: a single ReAct agent that chooses between tools instead of a linear “retrieve → answer” pipeline. The agent can:

- **RAG (mushtaq_docs)** — Answer from indexed portfolio data (projects, experience, skills).
- **GitHub (get_repo_tech_stack)** — Fetch live language/tech stack for Mushtaq’s repos.
- **Discord (notify_mushtaq)** — Send contact/hire requests to Mushtaq via Discord when the user provides name, message, and contact.

So it’s **one agent, one LLM, multiple tools** — with Multi-LLM support, and no longer a simple linear RAG flow.

🏗️ Architecture

    Framework: FastAPI (Python)
    RAG: LlamaIndex (Agentic — ReAct agent + RAG + tools)
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

```
├── app/
│   ├── __init__.py
│   ├── api.py          # API route definitions (/chat)
│   ├── rag_engine.py   # Agentic RAG: ReAct agent, index, tools wiring
│   ├── settings.py     # LLM selection & embed model config
│   └── tools.py        # GitHub tech-stack tool & Discord recruiter (notify_mushtaq)
├── data/               # Portfolio source documents (markdown)
├── scripts/            # Build index, test query (local)
├── storage/            # Persisted vector index files
├── main.py             # FastAPI entry point & Health Check
├── requirements.txt    # Project dependencies
├── Dockerfile          # Container config for Render
└── docker-compose.yml  # Local run with Docker
```


🛠️ Installation & Setup

    1. Clone the Repository
    https://github.com/MushtaqSoftDev/rag-portfolio-bot
    cd rag-portfolio-bot

    2. Environment Setup
    Create a .env file in the root directory:
    GOOGLE_API_KEY=your_gemini_api_key
    GROQ_API_KEY=your_groq_api_key
    DISCORD_WEBHOOK_URL=your_discord_webhook   # Optional: for recruiter/contact notifications
    MUSHTAQ_EMAIL=your_email                  # Optional: overrides contact email in hire/contact replies

    3. Install Dependencies
    pip install -r requirements.txt

    4. Run Locally
    python main.py