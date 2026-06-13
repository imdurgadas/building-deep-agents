# 05_parallel_executor.py
import concurrent.futures
import json
import time
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass, field

from child_agent import run_child_agent, FileReviewFindings
from workspace import read_plan, write_finding, update_task_status, read_all_findings

# Mock diff database — in production this comes from the harness's get_file_diff tool
MOCK_DIFFS: Dict[str, str] = {
    "src/auth/login.py": """@@ -10,18 +10,24 @@
 def login_user(request):
-    password_hash = md5(request.POST['password']).hexdigest()
-    query = "SELECT * FROM users WHERE username = '%s'" % request.POST['username']
-    user = db.execute(query).fetchone()
+    username = request.POST.get('username', '')
+    password = request.POST.get('password', '')
+    query = f"SELECT * FROM users WHERE username = '{username}'"
+    user = db.execute(query).fetchone()
     if user:
         return create_session(user)""",

    "src/auth/tokens.py": """@@ -5,6 +5,8 @@
 import jwt
-SECRET_KEY = "hardcoded-secret-123"
+SECRET_KEY = os.environ.get("JWT_SECRET", "hardcoded-secret-123")
+
 def create_token(user_id: int) -> str:
-    payload = {"user_id": user_id, "exp": time.time() + 3600}
+    payload = {"user_id": user_id, "exp": int(time.time()) + 3600}""",

    "src/db/user_repository.py": """@@ -22,6 +22,8 @@
 def get_users_with_orders():
+    # Naive implementation — N+1 query pattern
     users = db.query("SELECT * FROM users").fetchall()
+    for user in users:
+        user.orders = db.query(f"SELECT * FROM orders WHERE user_id = {user.id}").fetchall()
     return users""",

    "tests/test_auth.py": """@@ -1,8 +1,14 @@
 def test_login():
-    assert True  # TODO: implement
+    response = client.post("/login", data={"username": "test", "password": "test"})
+    assert response.status_code == 200""",
}

@dataclass
class TaskResult:
    """Result of executing a single review task."""
    task_id: str
    file_path: str
    review_type: str
    findings: Optional[FileReviewFindings] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0
    
    @property
    def succeeded(self) -> bool:
        return self.findings is not None and self.error is None

def execute_single_task(task: dict) -> TaskResult:
    """
    Execute a single review task. Designed to run in a thread pool.
    
    This function is the unit of work for each thread. It:
    1. Fetches the file diff (from mock or GitHub API)
    2. Runs the child agent
    3. Returns a TaskResult (success or failure — never raises)
    
    The never-raises contract is important: if this function raises,
    concurrent.futures will swallow the exception and the task silently disappears.
    By catching all exceptions and returning TaskResult(error=...), we ensure
    every task is accounted for in the final aggregation.
    """
    start_time = time.time()
    
    try:
        # Fetch diff content
        diff_content = MOCK_DIFFS.get(
            task["file_path"],
            f"# No diff available for {task['file_path']}\n+# Minor update"
        )
        
        # Run the child agent
        findings = run_child_agent(
            file_path=task["file_path"],
            diff_content=diff_content,
            review_type=task["review_type"]
        )
        
        return TaskResult(
            task_id=task["id"],
            file_path=task["file_path"],
            review_type=task["review_type"],
            findings=findings,
            duration_seconds=time.time() - start_time
        )
    
    except Exception as e:
        # Catch everything — return a structured failure rather than raising
        return TaskResult(
            task_id=task["id"],
            file_path=task["file_path"],
            review_type=task["review_type"],
            error=str(e),
            duration_seconds=time.time() - start_time
        )

def run_parallel_review(workspace_path: str, max_workers: int = 5) -> List[TaskResult]:
    """
    Run all pending review tasks in parallel using a thread pool.
    
    Args:
        workspace_path: Path to the PR workspace directory (contains plan.json)
        max_workers: Maximum concurrent child agents. Default 5.
    
    Why thread pool instead of asyncio?
    - LangChain's LLM clients are synchronous by default
    - ThreadPoolExecutor gives us true concurrency for I/O-bound work (LLM API calls)
    - asyncio would require an async-compatible LLM client throughout
    - For 5-20 concurrent files, threads are simple and effective
    
    Why max_workers=5 as default?
    - Most Google AI Studio free tier plans allow ~10 req/s
    - With 5 concurrent requests, each taking ~3s, we stay within rate limits
    - For paid tiers with higher rate limits, increase this to 10-15
    
    Returns:
        List of TaskResult objects (one per task, including failed ones)
    """
    workspace = Path(workspace_path)
    plan = read_plan(workspace)
    
    if not plan:
        raise ValueError(f"No plan found in workspace: {workspace_path}")
    
    # Filter to only pending tasks (supports resumability — already-completed tasks are skipped)
    pending_tasks = [t for t in plan["tasks"] if t["status"] == "pending"]
    pr_number = plan.get("pr_number", "unknown")
    
    print(f"\n🚀 Starting parallel review for PR #{pr_number}")
    print(f"   Total tasks: {len(plan['tasks'])} | Pending: {len(pending_tasks)} | Workers: {max_workers}")
    
    if not pending_tasks:
        print("   ✅ All tasks already completed. Nothing to do.")
        return []
    
    results = []
    total_start = time.time()
    
    # Update pending tasks to in_progress (for monitoring)
    for task in pending_tasks:
        update_task_status(workspace, task["id"], "in_progress")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the thread pool
        future_to_task = {
            executor.submit(execute_single_task, task): task
            for task in pending_tasks
        }
        
        print(f"\n⚡ {len(future_to_task)} tasks submitted to thread pool...")
        
        # Collect results as they complete (not in submission order)
        completed = 0
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            result = future.result()  # This won't raise — execute_single_task catches everything
            
            completed += 1
            progress = f"[{completed}/{len(pending_tasks)}]"
            
            if result.succeeded:
                print(f"\n  ✅ {progress} Completed: {result.file_path} ({result.duration_seconds:.1f}s)")
                print(f"         Risk: {result.findings.overall_risk.upper()} | Issues: {len(result.findings.issues)}")
                
                # Write findings to workspace
                write_finding(workspace, result.file_path, result.findings.model_dump())
                update_task_status(workspace, result.task_id, "completed",
                                 result=result.findings.summary)
            else:
                print(f"\n  ❌ {progress} Failed: {result.file_path}")
                print(f"         Error: {result.error}")
                update_task_status(workspace, result.task_id, "failed", result=result.error)
            
            results.append(result)
    
    total_duration = time.time() - total_start
    successful = sum(1 for r in results if r.succeeded)
    failed = sum(1 for r in results if not r.succeeded)
    
    print(f"\n📊 Parallel Review Complete")
    print(f"   Total time: {total_duration:.1f}s (vs {sum(r.duration_seconds for r in results):.1f}s sequential)")
    print(f"   Succeeded: {successful} | Failed: {failed}")
    
    return results



class ConsolidatedReview(BaseModel):
    """The final aggregated review report for the entire PR."""
    pr_number: int
    total_files_reviewed: int
    total_issues: int
    critical_issues: int
    high_issues: int
    pr_recommendation: Literal["block", "request_changes", "approve_with_notes", "approve"]
    files_by_risk: Dict[str, List[str]]  # {"critical": [...], "high": [...], ...}
    markdown_report: str

def aggregate_findings(workspace_path: str, pr_number: int) -> ConsolidatedReview:
    """
    Read all subagent findings from the workspace and produce a consolidated report.
    
    Aggregation logic:
    - PR recommendation is the worst finding across all files
    - Issues are counted and grouped by severity
    - Markdown report is formatted for GitHub PR comments
    """
    workspace = Path(workspace_path)
    all_findings = read_all_findings(workspace)
    
    if not all_findings:
        return ConsolidatedReview(
            pr_number=pr_number,
            total_files_reviewed=0,
            total_issues=0,
            critical_issues=0,
            high_issues=0,
            pr_recommendation="approve",
            files_by_risk={"critical": [], "high": [], "medium": [], "low": [], "none": []},
            markdown_report="No files were reviewed."
        )
    
    # Aggregate metrics
    total_issues = 0
    critical_issues = 0
    high_issues = 0
    files_by_risk: Dict[str, List[str]] = {
        "critical": [], "high": [], "medium": [], "low": [], "none": []
    }
    
    # Build markdown report sections
    report_lines = [
        f"# 🤖 DevPulse Automated Code Review\n",
        f"**PR #{pr_number}** | **Files Reviewed:** {len(all_findings)}\n",
        "---\n"
    ]
    
    # Determine overall recommendation (most severe wins)
    recommendation_priority = ["block", "request_changes", "approve_with_notes", "approve"]
    overall_recommendation = "approve"
    
    for finding_data in sorted(all_findings, key=lambda x: x.get("overall_risk", "none")):
        # Parse the stored finding
        file_path = finding_data.get("source_file", "unknown")
        overall_risk = finding_data.get("overall_risk", "none")
        issues = finding_data.get("issues", [])
        summary = finding_data.get("summary", "No summary available.")
        recommended_action = finding_data.get("recommended_action", "approve")
        
        # Update metrics
        total_issues += len(issues)
        for issue in issues:
            if issue.get("severity") == "critical":
                critical_issues += 1
            elif issue.get("severity") == "high":
                high_issues += 1
        
        files_by_risk.setdefault(overall_risk, []).append(file_path)
        
        # Update overall recommendation (take the worst one)
        if (recommendation_priority.index(recommended_action) <
                recommendation_priority.index(overall_recommendation)):
            overall_recommendation = recommended_action
        
        # Add to report
        risk_emoji = {
            "critical": "🔴 CRITICAL",
            "high": "🟠 HIGH",
            "medium": "🟡 MEDIUM",
            "low": "🟢 LOW",
            "none": "✅ CLEAN"
        }.get(overall_risk, "⚪ UNKNOWN")
        
        report_lines.append(f"### `{file_path}` — {risk_emoji}\n")
        report_lines.append(f"{summary}\n")
        
        if issues:
            report_lines.append("**Issues Found:**\n")
            for issue in issues:
                sev_badge = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(
                    issue.get("severity", "low"), "⚪"
                )
                line_ref = f" (line {issue['line']})" if issue.get("line") else ""
                report_lines.append(
                    f"- {sev_badge} **{issue.get('category', 'ISSUE')}**{line_ref}: "
                    f"{issue.get('description', '')}\n"
                    f"  *Fix: {issue.get('suggested_fix', 'N/A')}*\n"
                )
        
        report_lines.append("\n---\n")
    
    # Summary footer
    recommendation_display = {
        "block": "🛑 BLOCK — Do not merge. Critical issues must be resolved first.",
        "request_changes": "⚠️ REQUEST CHANGES — Non-blocking issues require attention.",
        "approve_with_notes": "📝 APPROVE WITH NOTES — Minor issues, but safe to merge.",
        "approve": "✅ APPROVED — No significant issues found."
    }.get(overall_recommendation, "⚪ UNKNOWN")
    
    report_lines.insert(2, f"**Overall Recommendation:** {recommendation_display}\n")
    report_lines.insert(3,
        f"**Issues:** {total_issues} total | {critical_issues} critical | {high_issues} high\n\n"
    )
    
    # Write the final report to workspace
    report_path = Path(workspace_path) / "final_review.md"
    with open(report_path, "w") as f:
        f.write("".join(report_lines))
    
    print(f"\n📋 Final report written to {report_path}")
    
    return ConsolidatedReview(
        pr_number=pr_number,
        total_files_reviewed=len(all_findings),
        total_issues=total_issues,
        critical_issues=critical_issues,
        high_issues=high_issues,
        pr_recommendation=overall_recommendation,
        files_by_risk=files_by_risk,
        markdown_report="".join(report_lines)
    )

if __name__ == "__main__":
    from planner import run_planning_phase
    
    # Step 1: Generate plan (from Part 1)
    sample_files = [
        "src/auth/login.py",
        "src/auth/tokens.py",
        "src/db/user_repository.py",
        "tests/test_auth.py"
    ]
    
    plan = run_planning_phase(
        pr_number=847,
        pr_title="Refactor authentication system with JWT token rotation",
        modified_files=sample_files
    )
    
    # Step 2: Run parallel review (this part)
    results = run_parallel_review(
        workspace_path=plan["workspace_path"],
        max_workers=3  # Conservative for API rate limits
    )
    
    # Step 3: Aggregate findings
    consolidated = aggregate_findings(plan["workspace_path"], pr_number=847)
    
    print(f"\n{'='*60}")
    print(f"PR #{consolidated.pr_number} Review Complete")
    print(f"{'='*60}")
    print(f"Files Reviewed: {consolidated.total_files_reviewed}")
    print(f"Total Issues: {consolidated.total_issues}")
    print(f"Critical: {consolidated.critical_issues} | High: {consolidated.high_issues}")
    print(f"Recommendation: {consolidated.pr_recommendation.upper()}")
    print(f"\nMarkdown Report Preview:")
    print(consolidated.markdown_report[:1000])
