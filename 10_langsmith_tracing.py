# 10_langsmith_tracing.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

def configure_langsmith(project_name: str = "DevPulse-Production", enable: bool = True) -> None:
    """
    Configure LangSmith tracing for the current process.
    
    Call this ONCE at startup, before any LangChain calls.
    All subsequent LangChain/LangGraph calls will be automatically traced.
    
    Args:
        project_name: The LangSmith project to log traces to
        enable: Set False to disable tracing (e.g., in test environments)
    
    Environment variables required (in .env):
        LANGSMITH_API_KEY: Your LangSmith API key from https://smith.langchain.com
    """
    if not enable:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        print("📊 [LangSmith] Tracing disabled")
        return
    
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    
    if not langsmith_key:
        print("⚠️  [LangSmith] LANGSMITH_API_KEY not set. Tracing will not be active.")
        print("    Get your API key at: https://smith.langchain.com")
        return
    
    # These environment variables activate automatic tracing
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_PROJECT"] = project_name
    
    print(f"📊 [LangSmith] Tracing enabled → Project: '{project_name}'")
    print(f"    View traces at: https://smith.langchain.com")

def add_run_metadata(pr_number: int, run_id: str, reviewer_id: str = "devpulse-bot") -> dict:
    """
    Create run metadata to attach to every LangSmith trace in this review.
    
    This metadata appears in the LangSmith UI and makes filtering much easier:
    - Filter all traces for a specific PR
    - See which runs were triggered by which reviewer
    - Track latency and cost per PR number
    
    Usage:
        config = {"metadata": add_run_metadata(847, "run_20260617_001")}
        chain.invoke(input, config=config)
    """
    return {
        "pr_number": str(pr_number),
        "run_id": run_id,
        "reviewer": reviewer_id,
        "environment": os.getenv("ENVIRONMENT", "development"),
        "devpulse_version": "1.0.0"
    }

def demonstrate_tracing():
    """
    Show how LangSmith captures a complete review chain trace.
    When you run this, open LangSmith and you'll see the full trace.
    """
    configure_langsmith(project_name="DevPulse-Demo", enable=True)
    
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    
    pr_number = 847
    run_id = "demo_run_001"
    metadata = add_run_metadata(pr_number, run_id)
    
    config = {"metadata": metadata, "tags": ["security_review", "demo"]}
    
    # This call will appear in LangSmith with the metadata attached
    response = llm.invoke(
        [
            SystemMessage(content="You are a security reviewer. Analyze the code diff for vulnerabilities."),
            HumanMessage(content="""
Review this diff for SQL injection:
```diff
-    query = "SELECT * FROM users WHERE id = " + user_id
+    query = f"SELECT * FROM users WHERE id = '{user_id}'"
```
""")
        ],
        config=config
    )
    
    print(f"\n📊 Review result:\n{response.content}")
    print(f"\n✅ Open LangSmith to see the full trace with metadata:")
    print(f"   Project: DevPulse-Demo | PR: #{pr_number} | Run: {run_id}")
