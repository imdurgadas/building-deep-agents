# 01_workspace.py
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

WORKSPACE_DIR = Path("./devpulse_workspace")

def init_workspace(pr_number: int) -> Path:
    """
    Create a fresh workspace directory for a PR review run.
    Returns the path to the workspace.
    """
    workspace = WORKSPACE_DIR / f"pr_{pr_number}"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "findings").mkdir(exist_ok=True)
    
    print(f"✅ Workspace initialized at: {workspace}")
    return workspace

def write_plan(workspace: Path, plan: dict) -> None:
    """
    Write the agent's task plan to the workspace.
    This is what allows the agent to resume if interrupted.
    """
    plan_path = workspace / "plan.json"
    plan["last_updated"] = datetime.utcnow().isoformat()
    
    with open(plan_path, "w") as f:
        json.dump(plan, f, indent=2)
    
    print(f"📝 Plan written to {plan_path}")

def read_plan(workspace: Path) -> Optional[dict]:
    """
    Read an existing plan from the workspace.
    Returns None if no plan exists (first run).
    """
    plan_path = workspace / "plan.json"
    if not plan_path.exists():
        return None
    
    with open(plan_path) as f:
        return json.load(f)

def update_task_status(workspace: Path, task_id: str, status: str, result: Optional[str] = None) -> None:
    """
    Update the status of a single task in the plan file.
    Statuses: 'pending' | 'in_progress' | 'completed' | 'failed'
    """
    plan = read_plan(workspace)
    if not plan:
        raise ValueError("No plan found. Cannot update task status.")
    
    for task in plan["tasks"]:
        if task["id"] == task_id:
            task["status"] = status
            if result:
                task["result_summary"] = result
            break
    
    write_plan(workspace, plan)
    print(f"🔄 Task '{task_id}' updated to status: {status}")

def write_finding(workspace: Path, file_path: str, findings: dict) -> None:
    """
    Write a subagent's review findings to its own file.
    Prevents different files' reviews from mixing in memory.
    """
    # Sanitize file path to create a valid filename
    safe_name = file_path.replace("/", "_").replace(".", "_") + ".json"
    finding_path = workspace / "findings" / safe_name
    
    findings["reviewed_at"] = datetime.utcnow().isoformat()
    findings["source_file"] = file_path
    
    with open(finding_path, "w") as f:
        json.dump(findings, f, indent=2)
    
    print(f"💾 Findings for '{file_path}' saved to workspace")

def read_all_findings(workspace: Path) -> list:
    """
    Read all subagent findings from the workspace findings directory.
    Used by the parent agent to aggregate results.
    """
    findings_dir = workspace / "findings"
    all_findings = []
    
    for finding_file in findings_dir.glob("*.json"):
        with open(finding_file) as f:
            all_findings.append(json.load(f))
    
    return all_findings
