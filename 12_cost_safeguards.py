# 12_cost_safeguards.py
import os
import time
import logging
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger("devpulse.cost")

@dataclass
class CostTracker:
    """
    Tracks API usage costs for a DevPulse review run.
    
    Pricing reference (approximate, as of 2026):
    - gemini-3.5-flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    - gemini-2.5-flash: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    - gemini-flash-latest: $0.075 per 1M input tokens, $0.30 per 1M output tokens
    """
    
    # Per-million-token pricing
    PRICING = {
        "gemini-3.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
        "gemini-flash-latest": {"input": 0.075, "output": 0.30},
    }
    
    max_budget_usd: float = 2.00  # Default: $2 per PR review
    current_cost_usd: float = field(default=0.0)
    input_tokens: int = field(default=0)
    output_tokens: int = field(default=0)
    tool_calls: int = field(default=0)
    start_time: datetime = field(default_factory=datetime.utcnow)
    
    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record tokens used by an LLM call and update cost estimate."""
        pricing = self.PRICING.get(model, self.PRICING["gemini-3.5-flash"])
        call_cost = (
            (input_tokens / 1_000_000) * pricing["input"] +
            (output_tokens / 1_000_000) * pricing["output"]
        )
        
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.current_cost_usd += call_cost
        
        logger.info(
            f"LLM call | model={model} | in={input_tokens} out={output_tokens} | "
            f"call_cost=${call_cost:.4f} | total=${self.current_cost_usd:.4f}"
        )
    
    def record_tool_call(self) -> None:
        self.tool_calls += 1
    
    def is_over_budget(self) -> bool:
        return self.current_cost_usd >= self.max_budget_usd
    
    def budget_remaining_usd(self) -> float:
        return max(0.0, self.max_budget_usd - self.current_cost_usd)
    
    def get_report(self) -> dict:
        elapsed = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            "total_cost_usd": f"${self.current_cost_usd:.4f}",
            "budget_usd": f"${self.max_budget_usd:.2f}",
            "budget_remaining_usd": f"${self.budget_remaining_usd():.4f}",
            "budget_used_pct": f"{self.current_cost_usd / self.max_budget_usd * 100:.1f}%",
            "input_tokens": f"{self.input_tokens:,}",
            "output_tokens": f"{self.output_tokens:,}",
            "tool_calls": self.tool_calls,
            "elapsed_seconds": f"{elapsed:.1f}s",
            "over_budget": self.is_over_budget()
        }

class IterationGuard:
    """
    Hard cap on reasoning iterations to prevent infinite agent loops.
    
    When an agent gets into a loop (e.g., calling the same tool repeatedly
    expecting a different result), the iteration guard stops execution
    before costs compound. Log the loop state for debugging.
    """
    
    def __init__(self, max_iterations: int = 15):
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self._tool_call_history: list = []
    
    def tick(self) -> None:
        """Call at the start of each iteration."""
        self.current_iteration += 1
    
    def record_tool_call(self, tool_name: str, args: dict) -> None:
        """Record a tool call to detect repetitive loops."""
        self._tool_call_history.append({
            "tool": tool_name,
            "args": str(args),
            "iteration": self.current_iteration
        })
    
    def should_stop(self) -> tuple[bool, str]:
        """
        Check whether the agent should stop.
        
        Returns (should_stop: bool, reason: str)
        """
        if self.current_iteration >= self.max_iterations:
            return True, f"Max iterations ({self.max_iterations}) reached"
        
        # Detect repetitive loops: same tool called with same args 3+ times
        if len(self._tool_call_history) >= 3:
            recent = self._tool_call_history[-3:]
            if (len(set(c["tool"] for c in recent)) == 1 and
                    len(set(c["args"] for c in recent)) == 1):
                tool = recent[0]["tool"]
                return True, f"Infinite loop detected: '{tool}' called 3x with identical args"
        
        return False, ""
    
    def assert_should_continue(self) -> None:
        """Raise an exception if the guard says to stop. Use in agent loops."""
        should_stop, reason = self.should_stop()
        if should_stop:
            raise RuntimeError(f"IterationGuard: {reason}")

# ---- Integrated Cost-Safe Execution Example ----

def execute_with_cost_safety(
    review_task: dict,
    max_budget_usd: float = 2.00,
    max_iterations: int = 10
) -> dict:
    """
    Execute a review task with full cost and iteration safeguards.
    """
    cost_tracker = CostTracker(max_budget_usd=max_budget_usd)
    iteration_guard = IterationGuard(max_iterations=max_iterations)
    
    print(f"\n💰 Starting cost-safe review with ${max_budget_usd:.2f} budget")
    print(f"🔄 Max iterations: {max_iterations}")
    
    try:
        for iteration in range(max_iterations + 1):
            iteration_guard.tick()
            
            # Check iteration guard
            should_stop, reason = iteration_guard.should_stop()
            if should_stop:
                logger.warning(f"Stopping review: {reason}")
                return {
                    "status": "stopped",
                    "reason": reason,
                    "cost_report": cost_tracker.get_report()
                }
            
            # Check cost budget
            if cost_tracker.is_over_budget():
                msg = f"Budget exhausted: ${cost_tracker.current_cost_usd:.4f} >= ${max_budget_usd:.2f}"
                logger.warning(msg)
                return {
                    "status": "budget_exhausted",
                    "reason": msg,
                    "cost_report": cost_tracker.get_report()
                }
            
            # Simulate an LLM call (in production: actual LLM invocation)
            print(f"\n  Iteration {iteration + 1}: Making LLM call...")
            simulated_input_tokens = 2400 + (iteration * 500)
            simulated_output_tokens = 350 + (iteration * 50)
            
            cost_tracker.record_llm_call(
                model="gemini-3.5-flash",
                input_tokens=simulated_input_tokens,
                output_tokens=simulated_output_tokens
            )
            
            print(f"  Cost so far: ${cost_tracker.current_cost_usd:.4f} / ${max_budget_usd:.2f}")
            
            # Simulate finding no tool calls on iteration 3 (agent finished)
            if iteration == 2:
                print(f"\n  ✅ Agent completed review.")
                return {
                    "status": "completed",
                    "cost_report": cost_tracker.get_report()
                }
    
    except RuntimeError as e:
        return {"status": "error", "reason": str(e), "cost_report": cost_tracker.get_report()}

if __name__ == "__main__":
    # Test cost safeguards
    result = execute_with_cost_safety(
        review_task={"file_path": "src/auth/login.py", "review_type": "security"},
        max_budget_usd=0.005,  # Very small budget to trigger safeguard quickly
        max_iterations=10
    )
    
    print(f"\n📊 Execution Result:")
    print(f"   Status: {result['status']}")
    print(f"   Cost Report: {result['cost_report']}")
