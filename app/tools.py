import requests
import os
from llama_index.core.tools import FunctionTool

# 1. GitHub Tech Stack Tool
def get_repo_tech_stack(repo_name: str):
    """Use this when the user asks which languages or technologies Mushtaq uses in a project. Fetches live language data from GitHub for repo_name (e.g. JavaFinchRobot, PythonFlask, PyTorch). Returns the list of languages used in that repo."""
    # Github URL
    url = f"https://api.github.com/repos/MushtaqSoftDev/{repo_name}/languages"
    response = requests.get(url)
    if response.status_code == 200:
        languages = response.json()
        # Simple logic the agent can use to 'categorize'
        return f"The project '{repo_name}' uses these technologies: {', '.join(languages.keys())}."
    return "Could not fetch details for this repository."

# Wrapping it as a tool
github_stack_tool = FunctionTool.from_defaults(fn=get_repo_tech_stack)

# 2. Discord Recruiter/Visitor tool
def notify_mushtaq(visitor_name: str, message: str, contact_info: str):
    """ONLY call when the user has EXPLICITLY given their name, a message (e.g. about the role), and contact (email/LinkedIn) in this conversation. Do NOT call if they only asked 'how to hire/contact' without providing those details. Sends a Discord notification to Mushtaq."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return "System Error: Discord Webhook not configured."

    # Guard: do not send if any field is missing or placeholder
    placeholders = ("unknown", "n/a", "na", "none", "not provided", "—", "-", "")
    v = (visitor_name or "").strip().lower()
    m = (message or "").strip().lower()
    c = (contact_info or "").strip().lower()
    if v in placeholders or m in placeholders or c in placeholders or not (visitor_name and message and contact_info):
        return "Cannot notify: need the visitor's name, message, and contact (email/LinkedIn) from their message. Ask them to share these, then try again."

    payload = {
        "content": (
            f"💼 **New Contact Request!**\n"
            f"**Name:** {visitor_name}\n"
            f"**Contact:** {contact_info}\n"
            f"**Message:** {message}"
        )
    }
    requests.post(webhook_url, json=payload)
    return "Notification sent successfully! Mushtaq will contact you soon."

recruiter_tool = FunctionTool.from_defaults(fn=notify_mushtaq)