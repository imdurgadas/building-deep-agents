# 11_hitl_approval.py
import sqlite3
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import json

class ApprovalWorkflowState(TypedDict):
    """State for a workflow that requires human approval before destructive actions."""
    pr_number: int
    proposed_jira_tickets: list[dict]  # Tickets the agent wants to create
    approved_tickets: list[dict]       # Tickets approved by human
    rejected_tickets: list[dict]       # Tickets rejected by human
    approval_requested: bool
    approval_granted: Optional[bool]
    final_report: str

def analyze_for_critical_issues(state: ApprovalWorkflowState) -> ApprovalWorkflowState:
    """
    Node 1: Analyze the PR and identify critical issues that require Jira tickets.
    This is the analysis phase — no external actions taken yet.
    """
    print(f"\n🔍 [Node: analyze] Analyzing PR #{state['pr_number']} for critical issues...")
    
    # In production: run the full subagent review from Part 3
    # For demo: return mock critical findings
    proposed_tickets = [
        {
            "title": "SQL Injection vulnerability in login_user()",
            "file": "src/auth/login.py",
            "line": 13,
            "priority": "CRITICAL",
            "description": "Raw SQL query with f-string interpolation. Immediate fix required.",
            "suggested_fix": "Use parameterized queries: cursor.execute('... WHERE id = ?', (user_id,))"
        },
        {
            "title": "Hardcoded JWT secret key detected",
            "file": "src/auth/tokens.py",
            "line": 6,
            "priority": "HIGH",
            "description": "JWT_SECRET falls back to hardcoded string when env var is missing.",
            "suggested_fix": "Remove the fallback: SECRET_KEY = os.environ['JWT_SECRET']  # will raise KeyError if not set"
        }
    ]
    
    print(f"   Found {len(proposed_tickets)} issues requiring Jira tickets:")
    for ticket in proposed_tickets:
        print(f"   {'🔴' if ticket['priority'] == 'CRITICAL' else '🟠'} [{ticket['priority']}] {ticket['title']}")
    
    return {
        "proposed_jira_tickets": proposed_tickets,
        "approval_requested": True
    }

def request_human_approval(state: ApprovalWorkflowState) -> ApprovalWorkflowState:
    """
    Node 2: Format the approval request for the human reviewer.
    Execution PAUSES after this node (interrupt_before=['create_tickets']).
    The human sees this output and can approve/reject before execution continues.
    """
    tickets = state["proposed_jira_tickets"]
    
    print(f"\n📢 [Node: request_approval] Preparing approval request for human review...")
    
    # Format a clear summary for the human
    approval_summary = {
        "action_required": "JIRA_TICKET_CREATION",
        "pr_number": state["pr_number"],
        "proposed_tickets": tickets,
        "instructions": (
            "DevPulse has identified the above issues and proposes to create Jira tickets. "
            "Review each ticket and approve or reject. "
            "To approve: app.update_state(config, {'approval_granted': True}, as_node='request_approval') "
            "To reject: app.update_state(config, {'approval_granted': False}, as_node='request_approval')"
        )
    }
    
    print(json.dumps(approval_summary, indent=2))
    
    return {}  # State already set from previous node; just pass through

def create_approved_tickets(state: ApprovalWorkflowState) -> ApprovalWorkflowState:
    """
    Node 3: Create the Jira tickets — BUT ONLY after human approval.
    This node runs after the interrupt_before pause.
    """
    if not state.get("approval_granted"):
        print("\n❌ [Node: create_tickets] Approval was denied. No tickets will be created.")
        return {
            "rejected_tickets": state["proposed_jira_tickets"],
            "approved_tickets": [],
            "final_report": "Ticket creation cancelled by reviewer."
        }
    
    print("\n✅ [Node: create_tickets] Approval granted. Creating Jira tickets...")
    
    created_tickets = []
    for ticket in state["proposed_jira_tickets"]:
        # In production: call create_jira_ticket from Part 2
        ticket_id = f"DP-{abs(hash(ticket['title'])) % 9000 + 1000}"
        print(f"   Created: {ticket_id} — {ticket['title']}")
        created_tickets.append({**ticket, "jira_id": ticket_id, "status": "created"})
    
    return {
        "approved_tickets": created_tickets,
        "rejected_tickets": [],
        "final_report": f"Created {len(created_tickets)} Jira ticket(s): {[t['jira_id'] for t in created_tickets]}"
    }

def build_approval_workflow():
    """
    Build the human-in-the-loop approval graph.
    
    The key: interrupt_before=["create_tickets"] tells LangGraph to
    PAUSE execution before the "create_tickets" node and wait for
    app.stream(None, config) to be called again after human input.
    """
    builder = StateGraph(ApprovalWorkflowState)
    
    builder.add_node("analyze", analyze_for_critical_issues)
    builder.add_node("request_approval", request_human_approval)
    builder.add_node("create_tickets", create_approved_tickets)
    
    builder.add_edge(START, "analyze")
    builder.add_edge("analyze", "request_approval")
    builder.add_edge("request_approval", "create_tickets")
    builder.add_edge("create_tickets", END)
    
    # SQLite checkpointer — required for interrupt_before to work
    # (The state must be persisted at the interrupt point)
    memory = SqliteSaver.from_conn_string(":memory:")
    
    return builder.compile(
        checkpointer=memory,
        interrupt_before=["create_tickets"]  # PAUSE here and wait for human input
    )

def run_approval_workflow_demo():
    """
    Demonstrates the complete HITL workflow:
    1. Run until the interrupt point
    2. Inspect proposed actions
    3. Provide approval (or rejection)
    4. Resume execution
    """
    app = build_approval_workflow()
    config = {"configurable": {"thread_id": "pr_847_approval_demo"}}
    
    initial_state: ApprovalWorkflowState = {
        "pr_number": 847,
        "proposed_jira_tickets": [],
        "approved_tickets": [],
        "rejected_tickets": [],
        "approval_requested": False,
        "approval_granted": None,
        "final_report": ""
    }
    
    print("=== Phase 1: Running analysis until interrupt point ===")
    
    # Run until interrupt_before["create_tickets"] — graph pauses here
    for event in app.stream(initial_state, config=config):
        if "__interrupt__" in event:
            print("\n⏸️  GRAPH PAUSED — Awaiting human approval")
            break
        node = list(event.keys())[0]
        print(f"   ✓ Node '{node}' completed")
    
    # Inspect current state (what the human sees)
    snapshot = app.get_state(config)
    proposed = snapshot.values.get("proposed_jira_tickets", [])
    
    print(f"\n=== Phase 2: Human review of {len(proposed)} proposed ticket(s) ===")
    for ticket in proposed:
        priority_icon = "🔴" if ticket["priority"] == "CRITICAL" else "🟠"
        print(f"\n  {priority_icon} [{ticket['priority']}] {ticket['title']}")
        print(f"     File: {ticket['file']} (line {ticket['line']})")
        print(f"     Fix: {ticket['suggested_fix']}")
    
    # Simulate human decision (in production: actual web UI interaction)
    human_decision = True  # Would come from UI/API in production
    print(f"\n  Human decision: {'APPROVED ✅' if human_decision else 'REJECTED ❌'}")
    
    print("\n=== Phase 3: Resuming with human approval ===")
    
    # Update state with human decision BEFORE resuming
    app.update_state(config, {"approval_granted": human_decision}, as_node="request_approval")
    
    # Resume from the interrupt point
    for event in app.stream(None, config=config):
        node = list(event.keys())[0]
        if node != "__end__":
            print(f"   ✓ Node '{node}' completed")
    
    # Final state
    final = app.get_state(config)
    print(f"\n📋 Final Report: {final.values.get('final_report', 'No report')}")

if __name__ == "__main__":
    run_approval_workflow_demo()
