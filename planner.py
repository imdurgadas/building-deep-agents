# 01_planner.py
import os
import json
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List, Literal
from pathlib import Path
from workspace import init_workspace, write_plan

load_dotenv()

# ---- Data Models ----

class ReviewTask(BaseModel):
    """A single task in the PR review plan."""
    id: str = Field(description="Unique identifier, e.g. 'review_auth_py'")
    description: str = Field(description="Clear description of what this task does")
    file_path: str = Field(description="The file path this task relates to")
    review_type: Literal["security", "performance", "style", "test_coverage"] = Field(
        description="The type of review to perform"
    )
    priority: Literal["critical", "high", "medium", "low"] = Field(
        description="Priority of this task"
    )
    status: Literal["pending", "in_progress", "completed", "failed"] = Field(
        default="pending"
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="IDs of tasks that must complete before this one"
    )

class PRReviewPlan(BaseModel):
    """Complete plan for reviewing a pull request."""
    pr_number: int
    pr_title: str
    total_files: int
    tasks: List[ReviewTask]
    estimated_duration_minutes: int = Field(
        description="Rough estimate of total runtime in minutes"
    )

# ---- LLM Setup ----

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    temperature=0,  # Deterministic output for planning — no creativity needed here
    max_retries=3
)

# ---- Planner ----

PLANNER_SYSTEM_PROMPT = """You are the DevPulse Planning Agent.

Your ONLY job is to produce a structured review plan for a given GitHub Pull Request.

Rules:
- Break the PR into individual file review tasks
- Assign the correct review_type based on file purpose:
  * src/auth/* → security (authentication logic, token handling)
  * src/db/* or *models* → security AND performance
  * tests/* → test_coverage
  * All others → style review by default
- Mark files containing authentication, payment, or database operations as CRITICAL or HIGH priority
- Identify task dependencies (e.g., you cannot aggregate results before all reviews are complete)
- Be conservative with estimates: 2 minutes per file on average

Output ONLY valid JSON matching the PRReviewPlan schema. No explanation."""

def generate_review_plan(pr_number: int, pr_title: str, modified_files: List[str]) -> PRReviewPlan:
    """
    Call the LLM to generate a structured review plan for the given PR.
    The plan is deterministic (temperature=0) and schema-validated.
    """
    
    file_list = "\n".join(f"- {f}" for f in modified_files)
    
    messages = [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=f"""
Plan a code review for this Pull Request:

PR Number: #{pr_number}
Title: {pr_title}
Modified Files ({len(modified_files)} total):
{file_list}

Generate a complete PRReviewPlan JSON object.
""")
    ]
    
    # Use structured output to get a type-safe Pydantic object back
    # This eliminates any chance of JSON parsing errors
    structured_llm = llm.with_structured_output(PRReviewPlan)
    plan = structured_llm.invoke(messages)
    
    print(f"\n✅ Planning complete. Generated {len(plan.tasks)} review tasks")
    print(f"⏱️  Estimated duration: {plan.estimated_duration_minutes} minutes")
    
    return plan

def run_planning_phase(pr_number: int, pr_title: str, modified_files: List[str]) -> dict:
    """
    Full planning phase: generates the plan and persists it to the workspace.
    Returns the workspace path for use by subsequent execution phases.
    """
    workspace = init_workspace(pr_number)
    
    print(f"\n🧠 Starting DevPulse planning phase for PR #{pr_number}...")
    print(f"   Files to review: {len(modified_files)}")
    
    # Generate the structured plan
    plan = generate_review_plan(pr_number, pr_title, modified_files)
    
    # Convert to dict and add workspace metadata before saving
    plan_dict = plan.model_dump()
    plan_dict["workspace_path"] = str(workspace)
    plan_dict["status"] = "planned"
    
    # Write to workspace — this is what enables resumability
    write_plan(workspace, plan_dict)
    
    # Print human-readable summary
    print("\n📋 Review Plan Summary:")
    for task in plan.tasks:
        priority_emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(task.priority, "⚪")
        print(f"  {priority_emoji} [{task.review_type.upper()}] {task.file_path}")
    
    return plan_dict

if __name__ == "__main__":
    # Simulate a realistic PR with mixed file types
    sample_pr_files = [
        "src/auth/login.py",
        "src/auth/tokens.py",
        "src/db/user_repository.py",
        "src/api/endpoints.py",
        "src/models/user.py",
        "tests/test_auth.py",
        "tests/test_api.py",
        "README.md"
    ]
    
    plan = run_planning_phase(
        pr_number=847,
        pr_title="Refactor authentication system with JWT token rotation",
        modified_files=sample_pr_files
    )
    
    print(f"\n🎯 Plan ready. Tasks: {len(plan['tasks'])}")
    print(f"📂 Workspace: {plan['workspace_path']}")
    print("\nNext step: Execute the review plan using the DevPulse Harness (Part 2)")
