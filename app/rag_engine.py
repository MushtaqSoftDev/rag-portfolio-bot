from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.readers.file import SimpleDirectoryReader

# Force settings to load (LLM selection happens here)
import app.settings  # noqa

STORAGE_DIR = Path("storage")
DATA_DIR = Path("data")


def load_or_build_index():
    if STORAGE_DIR.exists():
        print("🔄 Loading index from storage...")
        storage_context = StorageContext.from_defaults(
            persist_dir=str(STORAGE_DIR)
        )
        return load_index_from_storage(storage_context)

    print("🧠 Storage not found. Building index...")
    documents = SimpleDirectoryReader(str(DATA_DIR)).load_data()

    index = VectorStoreIndex.from_documents(documents)

    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    index.storage_context.persist(persist_dir=str(STORAGE_DIR))

    print("✅ Index built and persisted.")
    return index


# -------------------------
# Index + Retriever
# -------------------------
index = load_or_build_index()

retriever = VectorIndexRetriever(
    index=index,
    similarity_top_k=3,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
)


def ask_question(question: str) -> str:
    response = query_engine.query(question)
    return str(response)



