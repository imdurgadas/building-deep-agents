# 04_child_agent.py
import os
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# ---- Structured Output Models ----

class Issue(BaseModel):
    """A single code issue found during review."""
    line: Optional[int] = Field(default=None, description="Line number in the diff (if known).")
    category: str = Field(description="Issue category, e.g. SQL_INJECTION, N_PLUS_1_QUERY, HARDCODED_SECRET")
    description: str = Field(description="Clear, specific description of the issue.")
    severity: Literal["critical", "high", "medium", "low"]
    suggested_fix: str = Field(description="Concrete, actionable fix recommendation.")

class FileReviewFindings(BaseModel):
    """Complete findings from a child agent reviewing a single file."""
    file_path: str
    review_type: str
    issues: List[Issue] = Field(default_factory=list)
    overall_risk: Literal["critical", "high", "medium", "low", "none"] = Field(
        description="Highest severity across all found issues, or 'none' if no issues."
    )
    summary: str = Field(description="2-3 sentence plain-English summary of findings.")
    recommended_action: Literal["block", "request_changes", "approve_with_notes", "approve"] = Field(
        description="Overall PR action recommendation based on this file's findings."
    )

# ---- Review Type Prompts ----
# These are deliberately compact — every token in the system prompt
# is a token NOT available for the actual code diff being reviewed.

REVIEW_PROMPTS = {
    "security": """ROLE: Security Code Reviewer
FOCUS: OWASP Top 10 vulnerabilities ONLY.
CHECK FOR:
- SQL/NoSQL injection (raw queries, f-string interpolation in queries)
- Hardcoded secrets, API keys, or passwords
- Insecure authentication (weak hashing like MD5, plain-text comparison)
- Path traversal vulnerabilities
- Broken access control (missing auth checks)
IGNORE: Style, documentation, performance unless it has security implications.
SEVERITY RULES: SQL injection/secrets → critical. Auth bypass → high. Others → medium or low.""",

    "performance": """ROLE: Performance Code Reviewer
FOCUS: Performance anti-patterns ONLY.
CHECK FOR:
- N+1 query patterns (queries inside loops)
- Missing database indexes for frequently-filtered columns
- Blocking I/O in async functions (requests.get inside async def)
- Unbounded result sets (SELECT * without LIMIT on large tables)
- Unnecessary repeated computation (same calculation in a loop)
IGNORE: Style, documentation, security issues.
SEVERITY RULES: Blocking I/O in async → high. N+1 queries → medium. Others → low.""",

    "test_coverage": """ROLE: Test Coverage Reviewer
FOCUS: Test quality and coverage ONLY.
CHECK FOR:
- New functions/methods added with no corresponding tests
- Tests that only assert True with no real assertions
- Missing edge case tests (null inputs, empty collections, max values)
- Tests that mock so heavily they test nothing real
IGNORE: Style, performance, security.
SEVERITY RULES: Missing tests for auth/payment functions → high. Others → medium or low.""",

    "style": """ROLE: Code Style Reviewer
FOCUS: Code quality and maintainability ONLY.
CHECK FOR:
- Missing docstrings on public functions/classes
- Inconsistent naming (mixing snake_case and camelCase in Python)
- Dead code (commented-out blocks, unused imports)
- Functions exceeding 50 lines without clear decomposition
IGNORE: Security, performance, test coverage.
SEVERITY: All style issues are low or medium."""
}

# ---- Child Agent Factory ----

def create_child_agent_llm():
    """
    Child agents use a smaller, faster model than the parent coordinator.
    
    Rationale: child agents do focused, single-file analysis.
    They don't need the reasoning depth of the parent — they need speed.
    Using a cheaper model here keeps the overall cost of reviewing 23 files reasonable.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_retries=2,
        request_timeout=45
    )

def run_child_agent(file_path: str, diff_content: str, review_type: str) -> FileReviewFindings:
    """
    Run a single isolated child agent to review one file.
    
    This is the core unit of DevPulse's parallelism.
    Each call to this function is completely independent — it creates its own
    LLM instance, its own message history, and returns a typed findings object.
    
    Args:
        file_path: The file path being reviewed (used in findings metadata)
        diff_content: The git diff/patch content of the file
        review_type: One of 'security', 'performance', 'test_coverage', 'style'
    
    Returns:
        FileReviewFindings — a typed object the parent aggregator can process
    """
    llm = create_child_agent_llm()
    
    # Use structured output to guarantee a typed response
    structured_llm = llm.with_structured_output(FileReviewFindings)
    
    system_prompt = REVIEW_PROMPTS.get(review_type, REVIEW_PROMPTS["style"])
    
    # The child agent's entire context — deliberately minimal
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Review the following file diff:\n\n"
            f"**File:** `{file_path}`\n"
            f"**Review Type:** {review_type}\n\n"
            f"```diff\n{diff_content}\n```\n\n"
            f"Output a complete FileReviewFindings JSON object."
        ))
    ]
    
    print(f"  🔍 [Child Agent] Reviewing: {file_path} ({review_type})")
    
    findings = structured_llm.invoke(messages)
    
    print(f"  {'✅' if findings.overall_risk == 'none' else '⚠️'} [Child Agent] Done: {file_path} → {findings.overall_risk.upper()} risk")
    
    return findings
