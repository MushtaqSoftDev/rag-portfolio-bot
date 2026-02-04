from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever

# Force settings to load
import app.settings  # noqa

storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
)

def ask_question(question: str) -> str:
    response = query_engine.query(question)

    # ✅ Works for Groq, Gemini, HF, Mock
    return str(response)


