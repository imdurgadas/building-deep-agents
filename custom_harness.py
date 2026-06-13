# 12_custom_harness.py
from typing import Dict
from langchain_core.messages import SystemMessage

class DomainPromptLibrary:
    """
    A library of domain-specific, language-aware system prompts for DevPulse.
    
    These prompts are:
    - Compact (under 200 tokens) — following the WRITE strategy from Part 4
    - Language-specific — checking for language-specific vulnerability patterns
    - Review-type-specific — security, performance, coverage, or style
    """
    
    PROMPTS: Dict[str, Dict[str, str]] = {
        "python": {
            "security": """ROLE: Python Security Reviewer — DevPulse
FOCUS: Python-specific OWASP vulnerabilities ONLY.
CHECK:
- SQLAlchemy/Django ORM raw query usage (filter(id=user_input) is safe; execute(f"...") is not)
- MD5/SHA1 in hashlib for passwords (bcrypt/argon2 required)
- pickle.loads() or yaml.load() with untrusted input
- eval()/exec() with any variable input
- Path traversal: os.path.join() with user-controlled segments
- Django: missing @login_required, CSRF exempt abuse
IGNORE: Style, docstrings, type hints (unless they hide security bugs)
SEVERITY: pickle/eval with input → critical | Raw SQL → critical | Weak hash → high""",

            "performance": """ROLE: Python Performance Reviewer — DevPulse
FOCUS: Python-specific performance bottlenecks ONLY.
CHECK:
- Django ORM N+1: queries inside for loops without select_related/prefetch_related
- Synchronous requests.get() inside async def functions
- List comprehensions doing repeated function calls that could be cached
- Heavy computation inside Django view functions (should be in Celery tasks)
- Using + for string concatenation in loops (use str.join() or f-strings)
IGNORE: Security, style, test coverage
SEVERITY: Sync I/O in async → high | Django N+1 → medium | String concat in loop → low""",

            "test_coverage": """ROLE: Python Test Coverage Reviewer — DevPulse
FOCUS: Test quality for new Python code ONLY.
CHECK:
- New functions/methods added without corresponding test functions (def test_...)
- assert True or assert result (not None) with no real assertion
- Missing edge cases: None inputs, empty lists, zero values, max integer
- Django views with no test_client() calls
- Tests with mocked-out logic that test nothing real
SEVERITY: Missing tests for auth/payment functions → high | Others → medium"""
        },
        
        "go": {
            "security": """ROLE: Go Security Reviewer — DevPulse
FOCUS: Go-specific security vulnerabilities ONLY.
CHECK:
- database/sql: db.Query/Exec with fmt.Sprintf or string concatenation (use placeholders: ?, $1)
- crypto/md5 or crypto/sha1 for password hashing (use bcrypt or argon2id)
- net/http: missing input validation on Handler parameters
- os.Open() or ioutil.ReadFile() with user-controlled paths (path traversal)
- Goroutine-based processing of user data without bounds/timeout limits
IGNORE: Style, gofmt violations, performance
SEVERITY: SQL injection → critical | Path traversal → critical | Goroutine flooding → high""",

            "performance": """ROLE: Go Performance Reviewer — DevPulse
FOCUS: Go concurrency and performance issues ONLY.
CHECK:
- Goroutine leaks: go func() spawned without done channel or WaitGroup
- Channel operations without select and default (potential deadlock)
- sync.Mutex held across I/O operations (should use sync.RWMutex or channel)
- Unbounded goroutine creation in loops (use worker pool pattern)
- json.Unmarshal in hot paths (use streaming decoder for large payloads)
IGNORE: Security, style
SEVERITY: Goroutine leak → high | Deadlock potential → critical | Mutex over I/O → medium"""
        },
        
        "typescript": {
            "security": """ROLE: TypeScript/JavaScript Security Reviewer — DevPulse
FOCUS: Node.js/TypeScript security vulnerabilities ONLY.
CHECK:
- Prisma/Sequelize: raw() or $queryRaw() with template literals (SQL injection)
- jsonwebtoken: algorithm: 'none' or missing algorithm verification
- express: missing helmet(), CORS wildcards, no rate limiting
- eval() or new Function() with any variable content
- fs.readFile/writeFile with user-controlled paths
- Prototype pollution: Object.assign({}, userInput) or merge(target, userInput)
IGNORE: Style, unused imports, tsconfig strictness (covered by dedicated tool)
SEVERITY: SQL injection/eval → critical | JWT misconfig → high | Missing helmet → medium""",

            "performance": """ROLE: TypeScript Performance Reviewer — DevPulse
FOCUS: Node.js/TypeScript performance issues ONLY.
CHECK:
- await inside for loops (should use Promise.all() for parallel execution)
- Missing database connection pooling (new db.Client() per request)
- Synchronous fs operations (fs.readFileSync in request handlers)
- Missing pagination on database queries returning unlimited results
- Memory leaks: EventEmitter listeners added but never removed
SEVERITY: Sync I/O in handlers → high | await in loop → medium | Missing pagination → medium"""
        }
    }
    
    GENERIC_PROMPTS = {
        "security": "ROLE: Security Code Reviewer\nFOCUS: OWASP Top 10 vulnerabilities.\nSEVERITY: injection/secrets → critical | auth bypass → high | others → medium/low",
        "performance": "ROLE: Performance Reviewer\nFOCUS: N+1 queries, blocking I/O, unbounded loops.\nSEVERITY: blocking I/O → high | N+1 → medium",
        "test_coverage": "ROLE: Test Coverage Reviewer\nFOCUS: Missing tests, weak assertions.\nSEVERITY: missing critical tests → high",
        "style": "ROLE: Style Reviewer\nFOCUS: Naming, docs, dead code.\nSEVERITY: all style issues → low/medium"
    }
    
    def get_prompt(self, language: str, review_type: str) -> SystemMessage:
        """Get the most specific system prompt available for this language/review combination."""
        lang_prompts = self.PROMPTS.get(language, {})
        content = lang_prompts.get(review_type, self.GENERIC_PROMPTS.get(review_type, "Review the code."))
        return SystemMessage(content=content)


from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_core.tools import BaseTool
from pathlib import Path
from typing import Any
import time

from model_router import ModelRouter
from tool_registry import ToolRegistry
from workspace import write_finding, update_task_status, read_plan
from child_agent import FileReviewFindings
from parallel_executor import MOCK_DIFFS

class DomainAwareHarness:
    """
    The complete DevPulse domain harness — the final evolution of our agent system.
    
    For each review task, this harness:
    1. Detects the programming language from the file extension
    2. Selects the appropriate model via the ModelRouter (cost-optimized)
    3. Loads language-specific tools from the ToolRegistry
    4. Uses a language and review-type specific system prompt
    5. Runs the child agent with this precisely configured setup
    6. Returns typed findings for workspace persistence
    
    This is the culmination of all six parts:
    - File workspace (Part 1)
    - Pydantic validation + middleware (Part 2)
    - Isolated child agents (Part 3)
    - Compact system prompts (Part 4 — Write strategy)
    - Production checkpointing can wrap this (Part 5)
    """
    
    MAX_ITERATIONS = 8
    
    def __init__(self):
        self.router = ModelRouter()
        self.registry = ToolRegistry()
        self.prompt_library = DomainPromptLibrary()
        
        print("✅ DomainAwareHarness initialized")
        print(f"   Supported languages: {list(ToolRegistry.EXTENSION_MAP.values())}")
    
    def execute_task(self, task: dict, diff_content: str) -> FileReviewFindings:
        """
        Execute a single review task with full domain awareness.
        
        This is the complete, production-ready review execution for one file.
        """
        file_path = task["file_path"]
        review_type = task["review_type"]
        priority = task["priority"]
        
        # 1. Detect language
        language = self.registry.get_language(file_path)
        print(f"\n  📁 File: {file_path} | Language: {language} | Type: {review_type} | Priority: {priority}")
        
        # 2. Route to appropriate model
        llm = self.router.get_model(review_type, priority)
        
        # 3. Get language-specific tools
        domain_tools = self.registry.get_tools(file_path, review_type)
        
        # 4. Build structured output LLM (always use this for child agents)
        structured_llm = llm.with_structured_output(FileReviewFindings)
        
        # Bind domain-specific tools if any exist
        if domain_tools:
            llm_with_tools = llm.bind_tools(domain_tools)
        else:
            llm_with_tools = None
        
        # 5. Get domain-specific system prompt (the Write strategy)
        system_prompt = self.prompt_library.get_prompt(language, review_type)
        
        # 6. If we have tools, run a tool-calling loop first, then structured output
        tool_results_summary = ""
        if llm_with_tools and domain_tools:
            tool_results_summary = self._run_tool_phase(
                llm_with_tools, domain_tools, system_prompt, diff_content, file_path
            )
        
        # 7. Run structured output analysis (with tool results added to context if available)
        user_content = (
            f"Review the following code diff:\n\n"
            f"**File:** `{file_path}`\n"
            f"**Language:** {language}\n"
            f"**Review Type:** {review_type}\n\n"
            f"```diff\n{diff_content}\n```"
        )
        
        if tool_results_summary:
            user_content += f"\n\n**Static Analysis Tool Results:**\n{tool_results_summary}\n\n"
            user_content += "Incorporate the tool results into your structured analysis."
        
        messages = [
            system_prompt,
            HumanMessage(content=user_content + "\n\nOutput a complete FileReviewFindings JSON object.")
        ]
        
        findings = structured_llm.invoke(messages)
        
        print(f"  {'✅' if findings.overall_risk == 'none' else '⚠️ '} "
              f"Risk: {findings.overall_risk.upper()} | Issues: {len(findings.issues)}")
        
        return findings
    
    def _run_tool_phase(
        self,
        llm_with_tools,
        tools: list,
        system_prompt,
        diff_content: str,
        file_path: str
    ) -> str:
        """
        Run one tool-calling turn to collect static analysis results.
        Returns a summary string to include in the structured output prompt.
        """
        tool_map = {
            t.name if hasattr(t, "name") else t.__name__: t
            for t in tools
        }
        
        messages = [
            system_prompt,
            HumanMessage(content=(
                f"Run your static analysis tools on this code from `{file_path}`:\n\n"
                f"```\n{diff_content[:3000]}\n```"  # Truncate for tool phase
            ))
        ]
        
        tool_results = []
        
        for _ in range(3):  # Max 3 tool-calling turns
            response = llm_with_tools.invoke(messages)
            messages.append(response)
            
            if not response.tool_calls:
                break
            
            for tool_call in response.tool_calls:
                tool_fn = tool_map.get(tool_call["name"])
                if tool_fn:
                    try:
                        result = tool_fn.invoke(tool_call["args"])
                        tool_results.append(f"[{tool_call['name']}]: {result}")
                    except Exception as e:
                        tool_results.append(f"[{tool_call['name']}]: Error — {e}")
                    
                    messages.append(ToolMessage(
                        content=str(result),
                        tool_call_id=tool_call["id"]
                    ))
        
        return "\n\n".join(tool_results) if tool_results else ""

# ---- Full End-to-End Runner ----

def run_full_devpulse_review(pr_number: int, max_workers: int = 3):
    """
    Run the complete DevPulse review pipeline:
    1. Generate plan (Part 1 — workspace + planner)
    2. Execute parallel domain-aware reviews (Parts 2-4)
    3. Aggregate findings (Part 3)
    
    Checkpointing (Part 5) and HITL gates (Part 5) would wrap this
    in a production LangGraph graph.
    """
    import concurrent.futures
    from planner import run_planning_phase
    from workspace import read_plan, write_finding, update_task_status, read_all_findings
    
    # Sample PR (representing a polyglot codebase)
    sample_files = [
        "src/auth/login.py",          # Python + security
        "src/auth/tokens.py",          # Python + security
        "src/db/user_repository.py",   # Python + performance
        "api/handlers/user.go",        # Go + security
        "api/services/cache.go",       # Go + performance
        "frontend/src/api/client.ts",  # TypeScript + security
        "tests/test_auth.py",          # Python + test_coverage
    ]
    
    print(f"\n{'='*60}")
    print(f"🚀 DevPulse Full Review — PR #{pr_number}")
    print(f"{'='*60}")
    
    # Step 1: Generate plan
    plan = run_planning_phase(
        pr_number=pr_number,
        pr_title="Refactor auth system with polyglot microservices",
        modified_files=sample_files
    )
    
    workspace_path = plan["workspace_path"]
    workspace = Path(workspace_path)
    harness = DomainAwareHarness()
    
    # Step 2: Run domain-aware parallel reviews
    tasks = [t for t in plan["tasks"] if t["status"] == "pending"]
    
    # Add PR number to each task for context
    for task in tasks:
        task["pr_number"] = pr_number
    
    # Cost estimate before running
    cost_estimate = harness.router.estimate_cost_savings(tasks)
    print(f"\n💰 Cost Estimate:")
    print(f"   Without routing (all smart): {cost_estimate['uniform_smart_cost']}")
    print(f"   With routing: {cost_estimate['routing_cost']}")
    print(f"   Savings: {cost_estimate['savings_usd']} ({cost_estimate['savings_pct']})")
    
    results = []
    
    def execute_task_wrapper(task: dict):
        diff = MOCK_DIFFS.get(task["file_path"],
               f"@@ -1,3 +1,4 @@\n # {task['file_path']}\n+# Minor update")
        return task, harness.execute_task(task, diff)
    
    print(f"\n⚡ Running {len(tasks)} domain-aware reviews (workers: {max_workers})...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(execute_task_wrapper, task): task for task in tasks}
        
        for future in concurrent.futures.as_completed(futures):
            try:
                task, findings = future.result()
                write_finding(workspace, task["file_path"], findings.model_dump())
                update_task_status(workspace, task["id"], "completed", result=findings.summary)
                results.append((task, findings))
            except Exception as e:
                task = futures[future]
                update_task_status(workspace, task["id"], "failed", result=str(e))
                print(f"  ❌ Failed: {task['file_path']} — {e}")
    
    # Step 3: Aggregate
    all_findings = read_all_findings(workspace)
    total_issues = sum(len(f.get("issues", [])) for f in all_findings)
    critical = sum(1 for f in all_findings for i in f.get("issues", []) if i.get("severity") == "critical")
    high = sum(1 for f in all_findings for i in f.get("issues", []) if i.get("severity") == "high")
    
    recommendation = "block" if critical > 0 else ("request_changes" if high > 0 else "approve")
    
    print(f"\n{'='*60}")
    print(f"📊 DevPulse Review Complete — PR #{pr_number}")
    print(f"{'='*60}")
    print(f"Files reviewed: {len(all_findings)}")
    print(f"Total issues: {total_issues} ({critical} critical, {high} high)")
    print(f"Recommendation: {recommendation.upper()}")
    print(f"Workspace: {workspace_path}")

if __name__ == "__main__":
    run_full_devpulse_review(pr_number=847, max_workers=3)
