# 02_harness_setup.py
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

load_dotenv()

def build_resilient_llm() -> BaseChatModel:
    """
    Build a multi-model fallback chain for production resilience.
    
    Strategy:
    - Primary: gemini-2.0-flash (fast, cheap, low latency)
    - Fallback 1: gemini-1.5-pro (higher rate limits, longer context)
    - Fallback 2: gemini-1.5-flash (emergency fallback if both above fail)
    
    Why this order:
    - gemini-2.0-flash is the fastest and cheapest for most reasoning tasks
    - gemini-1.5-pro has better rate limits on the Tier 1 API plan
    - gemini-1.5-flash as last resort — still capable for most review tasks
    """
    primary = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature=0,
        max_retries=2,  # Retry twice before falling back
        request_timeout=30  # 30-second timeout per call
    )
    
    fallback_pro = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_retries=2,
        request_timeout=60  # More generous timeout for the larger model
    )
    
    fallback_flash = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0,
        max_retries=1,
        request_timeout=30
    )
    
    # Chain them together — if primary fails, fallback_pro is tried, then fallback_flash
    resilient_llm = primary.with_fallbacks(
        [fallback_pro, fallback_flash],
        exceptions_to_handle=(Exception,)  # Catch all exceptions for fallback
    )
    
    return resilient_llm


import time
import hashlib
import json
import logging
from typing import Callable, Any, Dict
from functools import wraps

logger = logging.getLogger("devpulse.middleware")

class ToolMiddlewareStack:
    """
    A composable middleware stack that wraps LangChain tools with:
    - In-memory caching (configurable TTL)
    - Global rate limiting (max N calls per window)
    - Structured audit logging
    - Execution timing metrics
    """
    
    def __init__(self, max_calls_per_window: int = 10, window_seconds: int = 60, cache_ttl_seconds: int = 300):
        self._cache: Dict[str, tuple[Any, float]] = {}  # key -> (result, timestamp)
        self._call_timestamps: list[float] = []
        self._max_calls = max_calls_per_window
        self._window_seconds = window_seconds
        self._cache_ttl = cache_ttl_seconds
    
    def _make_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate a stable cache key from function name and arguments."""
        payload = {"fn": func_name, "args": args, "kwargs": kwargs}
        # Use a hash so keys don't get unwieldy with large inputs
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
    
    def _is_cached(self, cache_key: str) -> bool:
        if cache_key not in self._cache:
            return False
        _, cached_at = self._cache[cache_key]
        return (time.time() - cached_at) < self._cache_ttl
    
    def _enforce_rate_limit(self, tool_name: str) -> None:
        """
        Sliding window rate limiter.
        Waits (blocks) if we have exceeded max_calls in the past window_seconds.
        """
        now = time.time()
        # Remove timestamps outside the current window
        self._call_timestamps = [
            t for t in self._call_timestamps
            if now - t < self._window_seconds
        ]
        
        if len(self._call_timestamps) >= self._max_calls:
            # Calculate how long to wait until the oldest call leaves the window
            oldest = self._call_timestamps[0]
            wait_time = self._window_seconds - (now - oldest) + 0.1
            logger.warning(
                f"[Rate Limit] Tool '{tool_name}' hit rate limit. "
                f"Throttling for {wait_time:.1f}s. "
                f"Current window: {len(self._call_timestamps)}/{self._max_calls} calls"
            )
            print(f"⏳ [Middleware] Rate limit hit. Waiting {wait_time:.1f}s before executing '{tool_name}'...")
            time.sleep(wait_time)
        
        self._call_timestamps.append(time.time())
    
    def wrap(self, tool_func: Callable) -> Callable:
        """
        Wrap a LangChain tool with the full middleware stack.
        Preserves all tool metadata needed by LangChain (name, docstring, schema).
        """
        
        @wraps(tool_func)
        def wrapped(*args, **kwargs):
            tool_name = tool_func.name if hasattr(tool_func, "name") else tool_func.__name__
            cache_key = self._make_cache_key(tool_name, args, kwargs)
            
            # --- Cache Check ---
            if self._is_cached(cache_key):
                cached_result, cached_at = self._cache[cache_key]
                age = int(time.time() - cached_at)
                logger.info(f"[Cache HIT] Tool '{tool_name}' | Age: {age}s | Key: {cache_key}")
                print(f"⚡ [Cache] Returning cached result for '{tool_name}' (age: {age}s)")
                return cached_result
            
            # --- Rate Limiting ---
            self._enforce_rate_limit(tool_name)
            
            # --- Pre-execution Audit Log ---
            start_time = time.time()
            logger.info(f"[TOOL CALL] '{tool_name}' | Args: {kwargs}")
            print(f"🔧 [Middleware] Executing tool: '{tool_name}'")
            
            try:
                result = tool_func(*args, **kwargs)
                
                # --- Post-execution Logging & Caching ---
                duration_ms = int((time.time() - start_time) * 1000)
                result_size = len(str(result))
                
                logger.info(
                    f"[TOOL SUCCESS] '{tool_name}' | "
                    f"Duration: {duration_ms}ms | "
                    f"Response size: {result_size} chars"
                )
                print(f"✅ [Middleware] '{tool_name}' completed in {duration_ms}ms ({result_size} chars)")
                
                # Cache the result
                self._cache[cache_key] = (result, time.time())
                
                return result
                
            except Exception as e:
                duration_ms = int((time.time() - start_time) * 1000)
                logger.error(f"[TOOL ERROR] '{tool_name}' | Duration: {duration_ms}ms | Error: {e}")
                print(f"❌ [Middleware] Tool '{tool_name}' failed: {e}")
                raise  # Re-raise so LangChain can feed the error back to the model
        
        # --- Preserve LangChain Tool Metadata ---
        # This is critical — LangChain uses these attributes to build tool schemas for the LLM
        if hasattr(tool_func, "name"):
            wrapped.name = tool_func.name
        if hasattr(tool_func, "description"):
            wrapped.description = tool_func.description
        if hasattr(tool_func, "args_schema"):
            wrapped.args_schema = tool_func.args_schema
        if hasattr(tool_func, "func"):
            wrapped.func = tool_func.func
        
        return wrapped


from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_core.tools import BaseTool
from typing import List, Any
import json

# Import our tools
from github_tools import get_file_diff, post_review_comment, create_jira_ticket

class DevPulseHarness:
    """
    The DevPulse agent harness — the production runtime for the code review agent.
    
    Responsibilities:
    - Manages the LLM with multi-model fallback
    - Applies middleware (caching, rate limiting, logging) to all tools
    - Manages the tool-call reasoning loop
    - Enforces max-iterations safety limit
    - Returns structured results for workspace persistence
    """
    
    MAX_ITERATIONS = 10  # Hard cap on reasoning turns to prevent infinite loops
    
    def __init__(self):
        # Build resilient LLM
        self.llm = build_resilient_llm()
        
        # Build middleware stack
        self.middleware = ToolMiddlewareStack(
            max_calls_per_window=20,  # Max 20 tool calls per minute
            window_seconds=60,
            cache_ttl_seconds=600    # Cache tool results for 10 minutes
        )
        
        # Wrap all tools with middleware
        self.tools = [
            self.middleware.wrap(get_file_diff),
            self.middleware.wrap(post_review_comment),
            self.middleware.wrap(create_jira_ticket)
        ]
        
        # Bind tools to the LLM — this injects tool schemas into every LLM call
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        
        # Build tool executor map for dispatching calls
        self._tool_map = {t.name if hasattr(t, "name") else t.__name__: t for t in self.tools}
        
        print("✅ DevPulse Harness initialized")
        print(f"   Tools registered: {list(self._tool_map.keys())}")
    
    def execute_review_task(self, task: dict) -> dict:
        """
        Execute a single review task from the PR plan.
        
        The task dict comes from the workspace plan.json and contains:
        - task.id, task.description, task.file_path, task.review_type, task.priority
        
        Returns a findings dict ready to be written to the workspace.
        """
        system_prompt = self._build_system_prompt(task)
        user_message = HumanMessage(content=(
            f"Review the file '{task['file_path']}' in pull request #{task.get('pr_number', 'unknown')}.\n"
            f"Focus on: {task['review_type']} issues.\n"
            f"Priority level: {task['priority']}.\n\n"
            f"Use the get_file_diff tool to fetch the code changes, then analyze them.\n"
            f"Use post_review_comment to report your findings directly on the PR.\n"
            f"If you find a CRITICAL security issue, also use create_jira_ticket."
        ))
        
        messages = [system_prompt, user_message]
        findings = {"task_id": task["id"], "issues_found": [], "severity": "none"}
        
        for iteration in range(self.MAX_ITERATIONS):
            response = self.llm_with_tools.invoke(messages)
            messages.append(response)
            
            # If no tool calls, the agent has finished reasoning
            if not response.tool_calls:
                findings["final_analysis"] = response.content
                findings["iterations_used"] = iteration + 1
                print(f"\n[Harness] Task '{task['id']}' completed in {iteration + 1} iteration(s)")
                break
            
            # Execute each requested tool call
            for tool_call in response.tool_calls:
                tool_result = self._execute_tool(tool_call)
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))
        else:
            # MAX_ITERATIONS reached — safety stop
            findings["final_analysis"] = "Max iterations reached. Partial review completed."
            findings["warning"] = f"Stopped at {self.MAX_ITERATIONS} iterations"
            logger.warning(f"Task '{task['id']}' hit max iteration limit of {self.MAX_ITERATIONS}")
        
        return findings
    
    def _execute_tool(self, tool_call: dict) -> Any:
        """Dispatch a tool call to the correct wrapped tool function."""
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        tool_func = self._tool_map.get(tool_name)
        if not tool_func:
            error_msg = f"Unknown tool: '{tool_name}'. Available: {list(self._tool_map.keys())}"
            logger.error(error_msg)
            return error_msg
        
        try:
            return tool_func(**tool_args)
        except Exception as e:
            error_msg = f"Tool '{tool_name}' failed: {str(e)}"
            logger.error(error_msg)
            return error_msg  # Return error as string so LLM can attempt recovery
    
    def _build_system_prompt(self, task: dict) -> SystemMessage:
        """
        Build a focused, task-specific system prompt.
        
        Key principle: the system prompt is minimal and directive, not conversational.
        Every word here uses context window tokens that could be used for actual code.
        Target: under 200 tokens.
        """
        review_type_instructions = {
            "security": (
                "Focus: OWASP Top 10 vulnerabilities.\n"
                "Specifically check for: SQL injection, hardcoded secrets, insecure authentication, "
                "path traversal, broken access control.\n"
                "Severity mapping: SQL injection/secrets = CRITICAL, auth issues = HIGH."
            ),
            "performance": (
                "Focus: N+1 queries, missing indexes, unbounded loops, blocking I/O in async context.\n"
                "Severity mapping: Blocking I/O = HIGH, N+1 queries = MEDIUM."
            ),
            "test_coverage": (
                "Focus: Missing test cases for new code paths, assert-only tests with no real assertions.\n"
                "Severity mapping: Missing tests for critical paths = HIGH."
            ),
            "style": (
                "Focus: Code style, naming conventions, missing docstrings.\n"
                "Severity: style issues are always LOW or INFO."
            )
        }
        
        instructions = review_type_instructions.get(task["review_type"], "Perform a general code review.")
        
        return SystemMessage(content=(
            f"ROLE: DevPulse {task['review_type'].title()} Reviewer\n"
            f"TASK: Review ONE file only. Do not comment on files not assigned to you.\n"
            f"{instructions}\n"
            f"FORMAT: For each issue found, call post_review_comment with specific line numbers.\n"
            f"STOP: When you have reviewed the diff and posted findings, stop immediately."
        ))

# ---- Entry Point ----

if __name__ == "__main__":
    harness = DevPulseHarness()
    
    # Simulate executing a single task from the plan
    sample_task = {
        "id": "review_auth_login_py_security",
        "description": "Security review of authentication login module",
        "file_path": "src/auth/login.py",
        "review_type": "security",
        "priority": "critical",
        "pr_number": 847
    }
    
    print("\n🚀 Running DevPulse Harness for a single task...")
    findings = harness.execute_review_task(sample_task)
    
    print(f"\n📊 Task Findings Summary:")
    print(f"   Issues found: {len(findings.get('issues_found', []))}")
    print(f"   Iterations used: {findings.get('iterations_used', 'N/A')}")
    print(f"\n   Final Analysis:\n   {findings.get('final_analysis', 'No analysis returned')[:300]}")
