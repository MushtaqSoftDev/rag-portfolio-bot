import torch
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.settings import Settings

print("CUDA available:", torch.cuda.is_available())

# -----------------------------
# Embeddings (CPU)
# -----------------------------
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# LLM (CAUSAL, CPU)
# -----------------------------
Settings.llm = HuggingFaceLLM(
    model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    tokenizer_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=256,
    generate_kwargs={"temperature": 0.1},
    model_kwargs={
        "dtype": torch.float32,
        "low_cpu_mem_usage": True
    }
)

# -----------------------------
# Load index
# -----------------------------
print("🔄 Loading index from storage...")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

query_engine = index.as_query_engine(
    similarity_top_k=3,
    response_mode="compact"
)

# -----------------------------
# CLI loop
# -----------------------------
while True:
    q = input("\n🧑 Ask a question (or 'exit'): ")
    if q.lower() == "exit":
        break

    response = query_engine.query(q)
    print("\n🤖 Answer:\n", response)

