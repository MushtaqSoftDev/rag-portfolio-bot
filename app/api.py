from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.rag_engine import ask_question
import traceback

router = APIRouter()

class ChatRequest(BaseModel):
    question: str

@router.post("/chat")
async def chat(req: ChatRequest):
    try:
        answer = ask_question(req.question)
        return {"answer": answer}

    except Exception as e:
        print("🔥 CHAT ERROR:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


