from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import router
import app.settings

app = FastAPI(title="RAG Portfolio Bot", version="1.0.0")

# --- ADD THIS PART ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------

app.include_router(router)

@app.get("/")
def health():
    return {"status": "ok"}
