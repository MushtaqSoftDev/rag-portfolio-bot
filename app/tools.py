import requests
import os
from llama_index.core.tools import FunctionTool

# 1. GitHub Tech Stack Tool
def get_repo_tech_stack(repo_name: str):
    """Fetches the languages and main files of a repo to determine if it is Front-end, Back-end or AI/ML."""
    # Github URL
    url = f"https://api.github.com/repos/MushtaqSoftDev/{repo_name}/languages"
    response = requests.get(url)
    if response.status_code == 200:
        languages = response.json()
        # Simple logic the agent can use to 'categorize'
        return f"The project '{repo_name}' uses these technologies: {', '.join(languages.keys())}."
    return "Could not fetch details fro this repository."

# Wrapping it as a tool
github_stack_tool = FunctionTool.from_defaults(fn=get_repo_tech_stack)

# 2. Discord Recruiter/Visitor tool
def notify_mushtaq(visitor_name: str, message: str, contact_info: str):
    """Sends a Discord notification to Mushta. ONLY call this after you have the user's name, message, and contact info."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        return "System Error: Discord Webhook not configured."
    
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