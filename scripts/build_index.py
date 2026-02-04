from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
from llama_index.core.storage import StorageContext
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

print("📄 Loading documents...")
documents = SimpleDirectoryReader("data").load_data()

print("🧠 Initializing CPU embedding model...")
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

print("🧠 Creating in-memory vector store...")
vector_store = SimpleVectorStore()
storage_context = StorageContext.from_defaults(vector_store=vector_store)

print("📦 Building index...")
index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model,
)

print("💾 Persisting index to ./storage ...")
storage_context.persist(persist_dir="storage")

print("✅ Index built successfully!")













