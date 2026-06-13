# 08_compress_strategy.py
from typing import List, Any, Tuple
from langchain_core.messages import (
    BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
)
from langchain_google_genai import ChatGoogleGenerativeAI

def estimate_message_tokens(messages: List[BaseMessage]) -> int:
    """Rough token estimation: 1 token ≈ 4 characters."""
    return sum(len(str(m.content)) for m in messages) // 4

def compress_message_history(
    messages: List[BaseMessage],
    token_budget: int = 8000,
    keep_last_n_turns: int = 2
) -> List[BaseMessage]:
    """
    Compress message history when it exceeds the token budget.
    
    Strategy:
    1. Always preserve: SystemMessage (the agent's identity and rules)
    2. Always preserve: The last N turns (most recent context)
    3. Compress: Everything in between with an LLM-generated summary
    
    Args:
        messages: The full message history
        token_budget: Token limit before compression triggers (default: 8,000)
        keep_last_n_turns: How many recent turns to preserve verbatim (default: 2)
    
    Returns:
        Compressed message list, ready to be passed to the next LLM call
    """
    current_tokens = estimate_message_tokens(messages)
    
    if current_tokens <= token_budget:
        return messages  # No compression needed
    
    print(f"⚠️  [Context] History at ~{current_tokens} tokens (budget: {token_budget}). Compressing...")
    
    # Separate message types
    system_messages = [m for m in messages if isinstance(m, SystemMessage)]
    non_system = [m for m in messages if not isinstance(m, SystemMessage)]
    
    if len(non_system) <= keep_last_n_turns * 2:
        # Too few messages to compress meaningfully
        return messages
    
    # Split: older messages to compress, recent messages to keep
    # Each "turn" is a pair: HumanMessage + AIMessage (+ optional ToolMessages)
    # We keep the last `keep_last_n_turns` pairs
    split_point = max(0, len(non_system) - (keep_last_n_turns * 2))
    messages_to_compress = non_system[:split_point]
    messages_to_keep = non_system[split_point:]
    
    # Generate a compact summary of the compressed messages
    summarizer = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0)
    
    history_text = "\n".join(
        f"{type(m).__name__}: {str(m.content)[:500]}"
        for m in messages_to_compress
    )
    
    summary_prompt = [HumanMessage(content=(
        f"Summarise the following agent conversation history in 3-5 sentences. "
        f"Focus on: what tasks were completed, what findings were made, "
        f"what files were reviewed, and what actions were taken. "
        f"Do NOT include what still needs to be done (that comes from the active task plan).\n\n"
        f"History:\n{history_text}"
    ))]
    
    summary_response = summarizer.invoke(summary_prompt)
    summary_text = summary_response.content
    
    compressed_history_message = SystemMessage(
        content=f"[COMPRESSED HISTORY] Prior work completed:\n{summary_text}"
    )
    
    compressed_tokens = estimate_message_tokens(
        system_messages + [compressed_history_message] + messages_to_keep
    )
    
    print(f"✅ [Context] Compressed from ~{current_tokens} to ~{compressed_tokens} tokens "
          f"({int((1 - compressed_tokens/current_tokens) * 100)}% reduction)")
    
    return system_messages + [compressed_history_message] + messages_to_keep



class ContextBudgetManager:
    """
    Manages the context budget for a running agent.
    Decides when to compress, when to offload, and when to stop.
    """
    
    # Token budget allocation for a 32k context window
    BUDGET_ALLOCATION = {
        "system_prompt": 0.05,      # 5%   → ~1,600 tokens
        "active_content": 0.50,     # 50%  → ~16,000 tokens (the file diff under review)
        "message_history": 0.25,    # 25%  → ~8,000 tokens
        "output_buffer": 0.20       # 20%  → ~6,400 tokens for model generation
    }
    
    def __init__(self, total_context_window: int = 32_000):
        self.total = total_context_window
        self.budgets = {
            k: int(v * total_context_window)
            for k, v in self.BUDGET_ALLOCATION.items()
        }
    
    def should_compress(self, messages: List[BaseMessage]) -> bool:
        """Returns True if message history has exceeded the history budget."""
        history_tokens = estimate_message_tokens(messages)
        return history_tokens > self.budgets["message_history"]
    
    def should_abort(self, messages: List[BaseMessage], active_content_tokens: int) -> bool:
        """
        Returns True if the total context load is dangerously high.
        This is the safety brake — if we're approaching the model's max context,
        we stop gracefully rather than getting a context overflow error.
        """
        history_tokens = estimate_message_tokens(messages)
        system_tokens = sum(
            estimate_message_tokens([m]) for m in messages
            if isinstance(m, SystemMessage)
        )
        total = history_tokens + active_content_tokens + system_tokens
        
        # Abort if we're using more than 80% of the total window
        # (leaving 20% for output buffer)
        threshold = int(self.total * 0.80)
        if total > threshold:
            print(f"🛑 [Context Budget] Context at {total}/{self.total} tokens. Aborting gracefully.")
            return True
        
        return False
    
    def get_compression_report(self, messages: List[BaseMessage]) -> dict:
        """Return a diagnostic report of current context usage."""
        history_tokens = estimate_message_tokens(messages)
        return {
            "history_tokens": history_tokens,
            "history_budget": self.budgets["message_history"],
            "history_usage_pct": int(history_tokens / self.budgets["message_history"] * 100),
            "should_compress": self.should_compress(messages),
            "budgets": self.budgets
        }


from pathlib import Path
import json

class WorkspaceOffloader:
    """
    Manages offloading of large intermediate results to workspace files.
    
    The key insight: an agent doesn't need to remember what it found in previous
    turns — it can always read the workspace file. Keeping findings in the message
    history wastes context; writing them to files preserves the information durably
    and keeps the context window clean.
    """
    
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
    
    def offload_analysis(self, filename: str, content: dict) -> str:
        """
        Write analysis results to a workspace file.
        Returns a compact reference string for the message history.
        """
        file_path = self.workspace / filename
        with open(file_path, "w") as f:
            json.dump(content, f, indent=2)
        
        # Return a compact reference — this is what goes into the message history
        # instead of the full content
        return (
            f"[Analysis saved to workspace: {filename}] "
            f"Key findings: {len(content.get('issues', []))} issues, "
            f"risk level: {content.get('overall_risk', 'unknown')}"
        )
    
    def load_analysis(self, filename: str) -> dict:
        """Load a previously offloaded analysis from the workspace."""
        file_path = self.workspace / filename
        if not file_path.exists():
            return {}
        with open(file_path) as f:
            return json.load(f)

# Usage example: demonstrating the full compress + offload workflow
def demonstrate_context_compression():
    """
    Shows a multi-turn agent conversation before and after compression.
    """
    
    # Simulate a growing message history (turn 8 of a 15-turn review)
    simulated_messages = [
        SystemMessage(content="ROLE: Security Reviewer. FOCUS: OWASP Top 10."),
        HumanMessage(content="Review PR #847. Start with src/auth/login.py"),
        AIMessage(content="I'll review login.py. Let me fetch the diff."),
        ToolMessage(content="@@ -10,15... [500 token diff]", tool_call_id="tc1"),
        AIMessage(content="Found SQL injection on line 13. Password uses MD5. Posting comment."),
        ToolMessage(content="Comment posted successfully.", tool_call_id="tc2"),
        HumanMessage(content="Now review src/auth/tokens.py"),
        AIMessage(content="Fetching tokens.py diff now."),
        ToolMessage(content="@@ -5,12... [400 token diff]", tool_call_id="tc3"),
        AIMessage(content="Found hardcoded JWT secret. Still using HS256 which is acceptable. Posting."),
        ToolMessage(content="Comment posted successfully.", tool_call_id="tc4"),
        HumanMessage(content="Now review src/db/user_repository.py"),
    ]
    
    budget_manager = ContextBudgetManager(total_context_window=32_000)
    report_before = budget_manager.get_compression_report(simulated_messages)
    
    print("=== Context Budget Report (Before Compression) ===")
    print(f"  History tokens: ~{report_before['history_tokens']}")
    print(f"  Budget: {report_before['history_budget']}")
    print(f"  Usage: {report_before['history_usage_pct']}%")
    print(f"  Should compress: {report_before['should_compress']}")
    
    # Apply compression
    compressed = compress_message_history(
        simulated_messages,
        token_budget=500,  # Low for demonstration
        keep_last_n_turns=1
    )
    
    report_after = budget_manager.get_compression_report(compressed)
    print(f"\n=== After Compression ===")
    print(f"  Messages: {len(simulated_messages)} → {len(compressed)}")
    print(f"  History tokens: ~{report_before['history_tokens']} → ~{report_after['history_tokens']}")
    
    # Show the compressed history message content
    for msg in compressed:
        if isinstance(msg, SystemMessage) and "COMPRESSED HISTORY" in str(msg.content):
            print(f"\n  Compressed summary:\n  {msg.content[:300]}")

if __name__ == "__main__":
    demonstrate_context_compression()


from workspace import read_plan, write_finding, update_task_status
from child_agent import run_child_agent
from select_strategy import get_file_diff, search_codebase
from pathlib import Path

class ContextEngineeredReviewer:
    """
    A reviewer that applies all four context engineering strategies:
    - WRITE: Uses compact, directive system prompts
    - SELECT: Loads files on-demand via tool calls
    - COMPRESS: Compresses parent coordinator history between files
    - ISOLATE: Each file review uses a fresh child agent context
    """
    
    def __init__(self, workspace_path: str):
        self.workspace = Path(workspace_path)
        self.budget_manager = ContextBudgetManager(total_context_window=32_000)
        self.offloader = WorkspaceOffloader(workspace_path)
        self.parent_messages = []  # Parent coordinator message history
    
    def run_review(self):
        """
        Run the full context-engineered review pipeline.
        """
        plan = read_plan(self.workspace)
        pending_tasks = [t for t in plan["tasks"] if t["status"] == "pending"]
        
        print(f"\n🧠 Starting Context-Engineered Review")
        print(f"   Pending tasks: {len(pending_tasks)}")
        print(f"   Context budget: {self.budget_manager.budgets}")
        
        for i, task in enumerate(pending_tasks):
            print(f"\n[{i+1}/{len(pending_tasks)}] Processing: {task['file_path']}")
            
            # STRATEGY 3: Compress parent history before each new file
            if self.budget_manager.should_compress(self.parent_messages):
                self.parent_messages = compress_message_history(
                    self.parent_messages,
                    token_budget=self.budget_manager.budgets["message_history"]
                )
            
            # STRATEGY 3 (abort check): Safety brake
            if self.budget_manager.should_abort(self.parent_messages, active_content_tokens=5000):
                print("⚠️  [Safety] Context budget exhausted. Stopping review.")
                update_task_status(self.workspace, task["id"], "failed",
                                 result="Stopped: context budget exhausted")
                break
            
            # STRATEGY 2 (SELECT): Fetch the diff on-demand (not upfront)
            # In production: use the get_file_diff tool from Part 2
            # For this demo: using mock data from Part 3
            from parallel_executor import MOCK_DIFFS
            diff_content = MOCK_DIFFS.get(
                task["file_path"],
                f"# No changes in {task['file_path']}"
            )
            
            # STRATEGY 4 (ISOLATE): Run fresh child agent with scoped context
            findings = run_child_agent(
                file_path=task["file_path"],
                diff_content=diff_content,
                review_type=task["review_type"]
            )
            
            # STRATEGY 3B (OFFLOAD): Save findings to file, not to parent messages
            compact_ref = self.offloader.offload_analysis(
                filename=f"findings_{task['id']}.json",
                content=findings.model_dump()
            )
            
            # Add only the compact reference to parent history (not full findings)
            self.parent_messages.append(
                AIMessage(content=f"Completed review of {task['file_path']}: {compact_ref}")
            )
            
            update_task_status(self.workspace, task["id"], "completed",
                             result=findings.summary)
        
        print(f"\n✅ Review complete. Context managed throughout.")
        print(f"   Final parent history: ~{estimate_message_tokens(self.parent_messages)} tokens")
