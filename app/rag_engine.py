from pathlib import Path

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    load_index_from_storage,
    PromptTemplate
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.readers import SimpleDirectoryReader

# Force settings to load (LLM selection happens here)
import app.settings  # noqa

# QueryEngine use default template that says=> answer the question based on context.
# But we want it domain specific only response related to portoflio

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

# Define a custom prompt
template = (
    "You are the AI Assistant for Mushtaq Portfolio. Your goal is to answer questions "
    "about Mushtaq projects, skills and experience based on hte provided context. \n"
    "-----------------\n"
    "{context_str}"
    "-----------------\n"
    "INSTRUCTIONS:\n"
    "- If the question is about Mushtaq, his work or projects, answer naturally. \n"
    "- If the question is NOT related to Mushtaq (e.g., 'How to bake a cake' or 'Who is the president'), "
    "polietly reply: 'I am Mushtaq protfolio assistant. I can only discuss his work and expertise. '\n"
    "- Do NOT use your own outside knowledge to answer off-topic questions. \n"
    "User Question: {query_str}\n"
    "Answer:"
)
qa_template = PromptTemplate(template)

# 2. Apply it to query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    text_qa_template=qa_template, 
)


def ask_question(question: str) -> str:
    response = query_engine.query(question)
    return str(response)



