from fastapi import FastAPI
from app.api import router
import app.settings  # 👈 IMPORTANT: forces LLM init

app = FastAPI(
    title="RAG Portfolio Bot",
    version="1.0.0"
)

app.include_router(router)

@app.get("/")
def health():
    return {"status": "ok"}
