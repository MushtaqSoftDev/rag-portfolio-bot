import os
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

# Contact email: show for hire/contact intents (configurable via env)
MUSHTAQ_EMAIL = os.getenv("MUSHTAQ_EMAIL", "mushtaquok70@gmail.com")

STRICT_SYSTEM_PROMPT = f"""You are Mushtaq's Portfolio AI Assistant.

    IDENTITY RULES:
    - If the user asks general knowledge (e.g., capital of a country, math, weather, history, science, or anything not about Mushtaq), refuse in a friendly way. Say something like: "I'm Mushtaq's portfolio assistant — I can only chat about his work, projects and expertise. What would you like to know about him?"
    - Do NOT use phrases like "I'm afraid" or "I don't have information". Keep the tone warm and helpful.
    - Do NOT answer any question that is not about Mushtaq. Refuse briefly and redirect.

    GITHUB TOOL RULES:
    - You have a tool 'get_repo_tech_stack(repo_name)' that fetches live language/tech data from GitHub.
    - When the user asks which languages or technologies Mushtaq uses in a project (e.g. "Java project", "Python project"), you MUST call get_repo_tech_stack with the repo name. Do NOT answer from memory or RAG only — use the tool for accurate, up-to-date tech stack.
    - Repo name mapping: Java project -> 'JavaFinchRobot', Python/Flask -> 'PythonFlask', AI/ML -> 'PyTorch', portfolio -> 'rag-portfolio-bot' or 'portfolio', Vue -> 'VueProject', NetSim -> 'NetSimulator', event scheduler -> 'event-scheduler', Master Informatica -> 'MasterInformatica'.
    - Known Repos: 'PyTorch', 'PythonFlask', 'rag-portfolio-bot', 'NetSimulator', 'event-scheduler', 'MasterInformatica', 'VueProject', 'JavaFinchRobot', 'portfolio'.

    HIRE / CONTACT RULES (IMPORTANT — Mushtaq is searching for jobs):
    - When the user asks how to hire/contact Mushtaq (e.g. "how can I hire mushtaq?") WITHOUT giving their own details, do NOT call notify_mushtaq. Instead reply with:
      1. Mushtaq's email: {MUSHTAQ_EMAIL}
      2. Say: "You can email him with details about the role, your contact info, and a short message. He'll get back to you."
      3. Optionally add: "If you share your name, company, and contact here, I can also notify him via Discord."
    - ONLY call 'notify_mushtaq(visitor_name, message, contact_info)' when the user has EXPLICITLY provided in this chat: their name (or company), a message (e.g. about the role/work), and contact (email or LinkedIn). If they only asked "how can I hire?" or "how to contact?" and did not give their name/message/contact, do NOT call the tool — just give the email and the instructions above.
    - When you do call notify_mushtaq, use real values from the user's message for visitor_name, message, and contact_info. Never use placeholders like "unknown" or "N/A".
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

# Fixed response for hire/contact so email and instructions are always shown (LLM often ignores prompt)
HIRE_CONTACT_RESPONSE = (
    f"You can reach Mushtaq at **{MUSHTAQ_EMAIL}**. "
    "Email him with details about the role, your contact info, and a short message — he'll get back to you. "
    "If you share your name, company, and contact here, I can also notify him via Discord."
)


def _is_hire_contact_only(question: str) -> bool:
    """True if the question is about hiring/contacting Mushtaq but doesn't provide recruiter details."""
    q = question.lower().strip()
    hire_contact = any(
        x in q for x in (
            "hire mushtaq", "hire him", "how to hire", "how can i hire",
            "contact mushtaq", "contact him", "how to contact", "how can i contact",
            "reach mushtaq", "get in touch", "recruit", "job for mushtaq",
            "want to hire", "looking to hire", "interested in hiring"
        )
    )
    if not hire_contact:
        return False
    # They're providing details if they include email, "my name", "i'm from", "contact me at", etc.
    has_details = (
        "@" in question or "my name is" in q or "i'm " in q or "im " in q
        or "contact me" in q or "reach me" in q or "linkedin" in q
        or "from " in q and ("company" in q or "recruiter" in q)
    )
    return not has_details


async def ask_question(question: str) -> str:
    # Guarantee hire/contact response includes email and "send details" (backend rule)
    if _is_hire_contact_only(question):
        return HIRE_CONTACT_RESPONSE

    # Reinforce scope in every request so the agent refuses off-topic questions
    instructional_query = (
        f"Remember: You are Mushtaq's Assistant. Only answer if the question is about Mushtaq (his work, projects, experience). "
        f"If the question is general knowledge or not about him, refuse and say you can only discuss Mushtaq. "
        f"Question: {question}"
    )
    handler = agent.run(user_msg=instructional_query)

    # Wait the handler get the actual result
    response = await handler

    return str(response)



