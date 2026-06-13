# 09_production_checkpointing.py
import sqlite3
from typing import TypedDict, List, Annotated, Optional
from pathlib import Path
from datetime import datetime

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import operator

load_dotenv()

# ---- Graph State ----
# Annotated[List[...], operator.add] is the LangGraph pattern for appended lists.
# Instead of replacing messages, new messages are APPENDED to the existing list.
# This is critical — without this annotation, a node returning {"messages": [...]}
# would REPLACE the entire message history, not add to it.

class DevPulseGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    pr_number: int
    current_task: Optional[dict]
    completed_tasks: Annotated[List[str], operator.add]
    failed_tasks: Annotated[List[str], operator.add]
    total_tokens_used: int
    max_token_budget: int
    run_id: str

# ---- Graph Nodes ----

def initialize_run(state: DevPulseGraphState) -> DevPulseGraphState:
    """
    Node 1: Initialize the review run.
    Sets up the initial state and marks the run as started.
    """
    print(f"\n🚀 [Node: initialize_run] Starting PR #{state['pr_number']} review")
    print(f"   Run ID: {state['run_id']}")
    print(f"   Token Budget: {state['max_token_budget']:,}")
    
    return {
        "messages": [SystemMessage(content=(
            f"DevPulse review started for PR #{state['pr_number']} "
            f"at {datetime.utcnow().isoformat()}Z"
        ))]
    }

def fetch_pr_tasks(state: DevPulseGraphState) -> DevPulseGraphState:
    """
    Node 2: Fetch the review tasks from the workspace plan.
    In production, this reads from the workspace plan.json created in Part 1.
    """
    # Mock task list for demonstration
    tasks = [
        {"id": "task_auth_login", "file_path": "src/auth/login.py", "review_type": "security", "priority": "critical"},
        {"id": "task_auth_tokens", "file_path": "src/auth/tokens.py", "review_type": "security", "priority": "high"},
        {"id": "task_db_repo", "file_path": "src/db/user_repository.py", "review_type": "performance", "priority": "high"},
    ]
    
    print(f"📋 [Node: fetch_pr_tasks] Loaded {len(tasks)} tasks")
    
    return {
        "messages": [AIMessage(content=f"Loaded {len(tasks)} review tasks from workspace plan.")],
        "current_task": tasks[0] if tasks else None
    }

def execute_review(state: DevPulseGraphState) -> DevPulseGraphState:
    """
    Node 3: Execute the current review task.
    
    This is where the child agent (from Part 3) would be invoked.
    The node updates both the message history and the completed_tasks list.
    
    IMPORTANT: If this node crashes mid-execution, the checkpointer
    has saved state at the END of the previous node. When we resume,
    we resume from the START of this node — meaning the task is retried
    from the beginning, not from the middle of the LLM call.
    """
    task = state["current_task"]
    if not task:
        return {"messages": [AIMessage(content="No current task. Review may be complete.")]}
    
    print(f"🔍 [Node: execute_review] Reviewing: {task['file_path']} ({task['review_type']})")
    
    # Token budget check (cost safeguard)
    if state["total_tokens_used"] >= state["max_token_budget"]:
        print(f"💰 [Node: execute_review] Token budget exhausted. Stopping.")
        return {
            "messages": [AIMessage(content=f"⚠️ Token budget exhausted. Stopping review.")],
            "failed_tasks": [task["id"]]
        }
    
    # Simulate the review (in production, call run_child_agent from Part 3)
    simulated_tokens = 2500
    simulated_finding = (
        f"Reviewed {task['file_path']}: "
        f"Found {'SQL injection vulnerability' if 'auth' in task['file_path'] else 'N+1 query pattern'}. "
        f"Severity: {task['priority']}. Comment posted to PR."
    )
    
    print(f"   ✅ Review complete. Findings: {simulated_finding[:80]}...")
    
    return {
        "messages": [AIMessage(content=simulated_finding)],
        "completed_tasks": [task["id"]],
        "total_tokens_used": state["total_tokens_used"] + simulated_tokens
    }

def check_completion(state: DevPulseGraphState) -> str:
    """
    Conditional edge: decide whether to continue reviewing or finish.
    Returns the name of the next node to execute.
    """
    # In production: check workspace plan for remaining pending tasks
    total_tasks = 3  # Mock total from our task list
    completed = len(state.get("completed_tasks", []))
    failed = len(state.get("failed_tasks", []))
    
    if completed + failed >= total_tasks:
        print(f"\n✅ [Conditional] All tasks done. Completed: {completed}, Failed: {failed}")
        return "aggregate"
    
    if state["total_tokens_used"] >= state["max_token_budget"]:
        print(f"\n💰 [Conditional] Budget exhausted. Moving to aggregate.")
        return "aggregate"
    
    print(f"  [Conditional] {completed}/{total_tasks} done. Continuing...")
    return "execute"  # Loop back to execute next task

def aggregate_results(state: DevPulseGraphState) -> DevPulseGraphState:
    """
    Node 4: Aggregate all findings and post the final review comment.
    """
    completed = state.get("completed_tasks", [])
    failed = state.get("failed_tasks", [])
    tokens_used = state.get("total_tokens_used", 0)
    
    summary = (
        f"PR #{state['pr_number']} review complete. "
        f"Tasks: {len(completed)} completed, {len(failed)} failed. "
        f"Tokens used: {tokens_used:,}/{state['max_token_budget']:,}. "
        f"[Mock] Final review comment posted to GitHub."
    )
    
    print(f"\n📊 [Node: aggregate_results] {summary}")
    
    return {
        "messages": [AIMessage(content=summary)]
    }

# ---- Graph Construction ----

def build_devpulse_graph(db_path: str = "devpulse_checkpoints.db"):
    """
    Build the DevPulse review graph with persistent SQLite checkpointing.
    
    The graph structure:
    START → initialize_run → fetch_pr_tasks → execute_review ↻ → aggregate_results → END
    
    The ↻ indicates a conditional loop: execute_review can route back to itself
    or forward to aggregate_results based on completion status.
    
    Why SQLite for checkpointing?
    - Zero-dependency persistence (SQLite is in Python's standard library)
    - Works locally and on any server with a persistent filesystem
    - For production Kubernetes deployments, switch to PostgresSaver to use
      a managed database that survives pod restarts
    """
    builder = StateGraph(DevPulseGraphState)
    
    # Add all nodes
    builder.add_node("initialize", initialize_run)
    builder.add_node("fetch_tasks", fetch_pr_tasks)
    builder.add_node("execute", execute_review)
    builder.add_node("aggregate", aggregate_results)
    
    # Static edges
    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "fetch_tasks")
    builder.add_edge("fetch_tasks", "execute")
    
    # Conditional loop edge — this is where the agent decides to continue or stop
    builder.add_conditional_edges(
        "execute",
        check_completion,
        {
            "execute": "execute",   # Loop: review next task
            "aggregate": "aggregate"  # Done: aggregate and post
        }
    )
    
    builder.add_edge("aggregate", END)
    
    # Attach the SQLite checkpointer
    conn = sqlite3.connect(db_path, check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # compile() creates a runnable graph with checkpointing enabled
    return builder.compile(checkpointer=checkpointer)

# ---- Execution with Resumability ----

def run_with_resumability(pr_number: int, run_id: str = None, budget: int = 50_000):
    """
    Run the DevPulse graph with full resumability support.
    
    The key is the `thread_id` in the config. Every run of the same PR
    uses the same thread_id. LangGraph stores all checkpoints under this ID.
    
    If the process crashes and you call this function again with the same
    thread_id, LangGraph automatically resumes from the last checkpoint —
    no code changes needed.
    
    Args:
        pr_number: The GitHub PR number to review
        run_id: Unique identifier for this run (defaults to pr_{pr_number})
        budget: Maximum tokens to use across all agent calls
    """
    import uuid
    
    if run_id is None:
        run_id = f"pr_{pr_number}"
    
    graph = build_devpulse_graph()
    
    # The thread_id is the resumability key
    # Same thread_id = same checkpoint stream = automatic resume on restart
    config = {
        "configurable": {
            "thread_id": run_id
        }
    }
    
    # Check if there's an existing checkpoint (resuming a previous run)
    existing_state = graph.get_state(config)
    if existing_state.values:
        completed = existing_state.values.get("completed_tasks", [])
        print(f"\n♻️  Resuming existing run '{run_id}'")
        print(f"   Previously completed tasks: {completed}")
        
        # Resume from checkpoint (pass None as input — LangGraph uses checkpoint state)
        for event in graph.stream(None, config=config):
            print(f"   Event: {list(event.keys())}")
    else:
        print(f"\n🆕 Starting new run '{run_id}'")
        
        # Initial state for a new run
        initial_state: DevPulseGraphState = {
            "messages": [],
            "pr_number": pr_number,
            "current_task": None,
            "completed_tasks": [],
            "failed_tasks": [],
            "total_tokens_used": 0,
            "max_token_budget": budget,
            "run_id": run_id
        }
        
        # Stream execution — each event is a node completion
        for event in graph.stream(initial_state, config=config):
            node_name = list(event.keys())[0]
            print(f"   ✓ Node '{node_name}' completed")
    
    # Final state
    final_state = graph.get_state(config)
    print(f"\n📊 Final State:")
    print(f"   Completed tasks: {final_state.values.get('completed_tasks', [])}")
    print(f"   Failed tasks: {final_state.values.get('failed_tasks', [])}")
    print(f"   Tokens used: {final_state.values.get('total_tokens_used', 0):,}")

if __name__ == "__main__":
    print("=== DevPulse Production Run with Checkpointing ===")
    run_with_resumability(pr_number=847, budget=50_000)
    
    print("\n\n=== Simulating Resume (same thread_id) ===")
    # In production, this would be called after a crash — same thread_id, picks up where left off
    run_with_resumability(pr_number=847, budget=50_000)
