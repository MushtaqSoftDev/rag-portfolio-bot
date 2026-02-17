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
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from app.tools import github_stack_tool, recruiter_tool  # Import new tools

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

STRICT_SYSTEM_PROMPT = """You are Mushtaq's Portfolio AI Assistant.

    IDENTITY RULES:
    - If the user asks general knowledge (e.g., 'Capital city of country', 'math', ''weater), you MUST refuse and say: "I am Mushtaq's portfolio assistant. I can discuss his work, projects and expertise."
    - Do NOT answer any question that is not about Mushtaq.

    GITHUB TOOL RULES:
    - You havea a tool 'get_repo_tech_stack'.
    - If a user asks for details about Mushtaq's AI/ML models or specific projects, you MUST use this tool.
    - Known Repos: 'PyTorch', 'PythonFlask', 'rag-portfolio-bot', 'NetSimulator', 'event-scheduler', 'MasterInformatica', 'VueProject', 'JavaFinchRobot', 'portfolio'.

    RECRUITER RULES:
    - If someone wants to contact Mushtaq, DO NOT use 'notify_mushtaq' until you have their Name, Company, and Contact Info.
    """

# agent_system_prompt = PromptTemplate(STRICT_SYSTEM_PROMPT)

# 2. Apply it to query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever=retriever,
    #text_qa_template=qa_template, 
)

# 3. Transforming linear to Agentic RAG
# 3.1 Wrapping the RAG as a tool
rag_tool = QueryEngineTool(
    query_engine=query_engine,
    metadata=ToolMetadata(
        name="mushtaq_docs",
        description="Information about Mushtaq projects, experience and project descriptions."
    ),
)

# 3.2 Create the ReAct Agent
# The agent will now choose b/w: RAG, Github or Discord
agent = ReActAgent(
    name="MushtaqPortfolioAgent",
    tools=[rag_tool, github_stack_tool, recruiter_tool],
    llm=app.settings.Settings.llm,
    verbose=True, #Setting this to true to see Render Logs
    context=STRICT_SYSTEM_PROMPT,
    max_iterations=10 # Safety limit for reasoning loops
)




# 3.3 Update the ask_question function

async def ask_question(question: str) -> str:
    # Using agent.chat instead of query_engine.query
    #response = query_engine.query(question)
    instructional_query = (
        f"Remember: You are Mushtaq's Assistant. Only answer if relevant to him. "
        f"Question: {question}"
    )
     
    handler = agent.run(user_msg=question)

    # Wait the handler get the actual result
    response = await handler

    return str(response)



