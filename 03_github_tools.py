# 03_github_tools.py
import os
import json
import requests
from pydantic import BaseModel, Field, field_validator
from langchain_core.tools import tool
from typing import Optional, Literal

# ---- Input Schemas ----

class GetFileDiffSchema(BaseModel):
    """Schema for fetching a file's diff from a GitHub PR."""
    pr_number: int = Field(
        ...,
        gt=0,
        description="The GitHub pull request number. Must be a positive integer."
    )
    file_path: str = Field(
        ...,
        min_length=1,
        description="The relative file path inside the repository, e.g. 'src/auth/login.py'."
    )
    
    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v: str) -> str:
        # Prevent path traversal attacks
        if ".." in v or v.startswith("/"):
            raise ValueError("file_path must be a relative path without '..' traversal")
        return v

class PostReviewCommentSchema(BaseModel):
    """Schema for posting a structured review comment to a GitHub PR."""
    pr_number: int = Field(..., gt=0, description="The pull request number.")
    body: str = Field(
        ...,
        min_length=10,
        description="Markdown content of the review comment. Minimum 10 characters."
    )
    file_path: Optional[str] = Field(
        default=None,
        description="Optional file path for line-specific inline comments."
    )
    line: Optional[int] = Field(
        default=None,
        gt=0,
        description="Optional line number for inline code review comments."
    )
    severity: Literal["info", "warning", "error"] = Field(
        default="info",
        description="Severity level that controls emoji prefix in the comment."
    )

class CreateJiraTicketSchema(BaseModel):
    """Schema for creating a Jira ticket for critical issues found during review."""
    title: str = Field(
        ...,
        min_length=5,
        max_length=200,
        description="Short, descriptive title for the issue."
    )
    description: str = Field(
        ...,
        min_length=20,
        description="Detailed description including file path, line reference, and suggested fix."
    )
    priority: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"] = Field(
        description="Issue priority. Use CRITICAL only for security vulnerabilities or data loss risks."
    )
    affected_files: list[str] = Field(
        default_factory=list,
        description="List of file paths affected by this issue."
    )

# ---- Tool Implementations ----

SEVERITY_EMOJI = {"info": "ℹ️", "warning": "⚠️", "error": "🔴"}

@tool(args_schema=GetFileDiffSchema)
def get_file_diff(pr_number: int, file_path: str) -> str:
    """
    Fetch the patch diff content of a specific file in a GitHub pull request.
    Returns the unified diff format showing what lines were added and removed.
    """
    token = os.getenv("GITHUB_TOKEN")
    
    if not token or token.startswith("your_"):
        # Development mock — returns realistic-looking diffs for testing
        return _mock_file_diff(file_path)
    
    # Real GitHub API call
    repo = os.getenv("GITHUB_REPO", "your-org/devpulse-demo")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    
    files = response.json()
    for file in files:
        if file["filename"] == file_path:
            return file.get("patch", "[Binary file or no diff available]")
    
    return f"[File '{file_path}' not found in PR #{pr_number} diff]"

@tool(args_schema=PostReviewCommentSchema)
def post_review_comment(
    pr_number: int,
    body: str,
    file_path: Optional[str] = None,
    line: Optional[int] = None,
    severity: str = "info"
) -> str:
    """
    Post a structured review comment or inline annotation to a GitHub pull request.
    For critical security issues, include the severity=error parameter.
    """
    emoji = SEVERITY_EMOJI.get(severity, "ℹ️")
    formatted_body = f"{emoji} **DevPulse Review**\n\n{body}"
    
    token = os.getenv("GITHUB_TOKEN")
    if not token or token.startswith("your_"):
        # Development mock — print what would be posted
        location = f"`{file_path}` line {line}" if file_path else "general PR comment"
        print(f"\n[GitHub Mock] Would post {severity} comment to PR #{pr_number} at {location}:")
        print(f"  {formatted_body[:200]}{'...' if len(body) > 200 else ''}")
        return f"Mock: Comment posted successfully to PR #{pr_number}"
    
    repo = os.getenv("GITHUB_REPO", "your-org/devpulse-demo")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3.json"
    }
    
    if file_path and line:
        # Post inline review comment on a specific line
        url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/comments"
        payload = {
            "body": formatted_body,
            "path": file_path,
            "line": line,
            "side": "RIGHT"  # Comment on the new version of the file
        }
    else:
        # Post general PR review comment
        url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
        payload = {"body": formatted_body}
    
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    return f"Comment posted successfully. Comment ID: {response.json().get('id')}"

@tool(args_schema=CreateJiraTicketSchema)
def create_jira_ticket(
    title: str,
    description: str,
    priority: str,
    affected_files: list[str] = None
) -> str:
    """
    Create a Jira ticket in the DevPulse project for critical issues found during code review.
    Only use this tool for issues that require developer action beyond the current PR.
    """
    files_str = "\n".join(f"- {f}" for f in (affected_files or []))
    full_description = f"{description}\n\n**Affected Files:**\n{files_str}"
    
    jira_base = os.getenv("JIRA_BASE_URL")
    jira_token = os.getenv("JIRA_API_TOKEN")
    
    if not jira_base or not jira_token:
        ticket_id = f"DP-{abs(hash(title)) % 9000 + 1000}"
        print(f"\n[Jira Mock] Would create ticket:")
        print(f"  ID: {ticket_id}")
        print(f"  Priority: {priority}")
        print(f"  Title: {title}")
        return f"Mock: Jira ticket {ticket_id} created with priority {priority}"
    
    # Real Jira API call (Cloud REST API v3)
    url = f"{jira_base}/rest/api/3/issue"
    headers = {
        "Authorization": f"Bearer {jira_token}",
        "Content-Type": "application/json"
    }
    priority_map = {"CRITICAL": "Highest", "HIGH": "High", "MEDIUM": "Medium", "LOW": "Low"}
    payload = {
        "fields": {
            "project": {"key": os.getenv("JIRA_PROJECT_KEY", "DEVPULSE")},
            "summary": title,
            "description": {
                "type": "doc",
                "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": full_description}]}]
            },
            "issuetype": {"name": "Bug"},
            "priority": {"name": priority_map.get(priority, "Medium")}
        }
    }
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    response.raise_for_status()
    ticket_key = response.json()["key"]
    return f"Jira ticket created: {ticket_key}"

# ---- Mock Data for Development ----

def _mock_file_diff(file_path: str) -> str:
    """Return realistic mock diffs for development and testing."""
    mocks = {
        "src/auth/login.py": """@@ -10,18 +10,24 @@
 def login_user(request):
-    password_hash = md5(request.POST['password']).hexdigest()
-    query = "SELECT * FROM users WHERE username = '%s'" % request.POST['username']
-    user = db.execute(query).fetchone()
+    username = request.POST.get('username', '')
+    password = request.POST.get('password', '')
+    # TODO: This is still using raw string interpolation — SQL injection risk!
+    query = f"SELECT * FROM users WHERE username = '{username}'"
+    user = db.execute(query).fetchone()
     if user and password_hash == user.password:
         return create_session(user)
     return None""",
        "src/auth/tokens.py": """@@ -5,12 +5,15 @@
 import jwt
 import time
-SECRET_KEY = "hardcoded-secret-do-not-use"
+SECRET_KEY = os.environ.get("JWT_SECRET", "hardcoded-secret-do-not-use")
 
 def create_token(user_id: int) -> str:
-    payload = {"user_id": user_id, "exp": time.time() + 3600}
+    payload = {
+        "user_id": user_id,
+        "exp": int(time.time()) + 3600,
+        "iat": int(time.time())
+    }
     return jwt.encode(payload, SECRET_KEY, algorithm="HS256")""",
        "src/db/user_repository.py": """@@ -22,8 +22,12 @@
 def get_user_by_email(email: str):
-    return db.query(f"SELECT * FROM users WHERE email = '{email}'")
+    # Partially fixed but still vulnerable if email contains special chars
+    sanitized = email.replace("'", "''")
+    return db.query(f"SELECT * FROM users WHERE email = '{sanitized}'")\
+        .fetchone()"""
    }
    
    return mocks.get(file_path, f"@@ -1,3 +1,4 @@\n # {file_path}\n+# Minor formatting update\n unchanged line\n")
