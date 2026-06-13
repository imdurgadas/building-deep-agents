# 13_model_router.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
from typing import Literal
import os

ReviewType = Literal["security", "performance", "test_coverage", "style"]
Priority = Literal["critical", "high", "medium", "low"]

class ModelRouter:
    """
    Routes review tasks to the most appropriate LLM based on:
    - Task type (security audits need stronger reasoning)
    - Priority (critical tasks warrant the most capable model)
    - Cost optimization (cheap tasks get cheap models)
    
    Model selection logic:
    ┌──────────────────────────────────────────────────────────────┐
    │ CRITICAL security  → gemini-2.5-pro  (best reasoning depth) │
    │ HIGH security/perf → gemini-3.5-flash (fast + capable)      │  
    │ MEDIUM security    → gemini-3.5-flash                       │
    │ Style/test checks  → gemini-3.5-flash (no deep reasoning)   │
    └──────────────────────────────────────────────────────────────┘
    
    Cost impact: routing correctly can reduce per-review costs by 60-70%
    because style/coverage tasks (which are the majority) use the cheaper model.
    """
    
    def __init__(self):
        # Model instances — created once, reused across tasks
        self._models = {
            "fast": ChatGoogleGenerativeAI(
                model="gemini-3.5-flash",
                temperature=0,
                max_retries=2
            ),
            "smart": ChatGoogleGenerativeAI(
                model="gemini-2.5-pro",
                temperature=0,
                max_retries=2
            )
        }
        
        # Routing table: (review_type, priority) → model_key
        # Logic: only route to 'smart' when both the task type AND priority warrant it
        self._routing_table = {
            ("security", "critical"): "smart",
            ("security", "high"):     "smart",  # Security + high → always smart
            ("security", "medium"):   "fast",   # Security + medium → fast is fine
            ("security", "low"):      "fast",
            ("performance", "critical"): "smart",
            ("performance", "high"):     "fast",
            ("performance", "medium"):   "fast",
            ("performance", "low"):      "fast",
            ("test_coverage", "critical"): "fast",
            ("test_coverage", "high"):     "fast",
            ("test_coverage", "medium"):   "fast",
            ("test_coverage", "low"):      "fast",
            ("style", "critical"):   "fast",
            ("style", "high"):       "fast",
            ("style", "medium"):     "fast",
            ("style", "low"):        "fast",
        }
    
    def get_model(self, review_type: ReviewType, priority: Priority) -> BaseChatModel:
        """
        Select and return the appropriate model for the task.
        
        Returns the model instance with multi-model fallback.
        The 'smart' model falls back to 'fast'; the 'fast' model falls back to
        the 'smart' model (as a last resort to avoid complete failure).
        """
        model_key = self._routing_table.get((review_type, priority), "fast")
        selected = self._models[model_key]
        fallback = self._models["smart" if model_key == "fast" else "fast"]
        
        routing_reason = self._get_routing_reason(review_type, priority, model_key)
        print(f"  🧠 [Router] {review_type}/{priority} → {model_key} model | Reason: {routing_reason}")
        
        return selected.with_fallbacks([fallback])
    
    def _get_routing_reason(self, review_type: str, priority: str, model_key: str) -> str:
        if model_key == "smart":
            return f"{priority} {review_type} requires deep reasoning and high accuracy"
        elif review_type == "style":
            return "Style checks do not require deep reasoning — fast model is sufficient"
        elif review_type == "test_coverage":
            return "Coverage analysis is pattern-matching — fast model handles it well"
        else:
            return f"{priority} {review_type} — fast model provides adequate analysis"
    
    def estimate_cost_savings(self, tasks: list) -> dict:
        """
        Estimate cost savings from intelligent routing vs. using the smart model for everything.
        """
        # Approximate token costs per task
        SMART_COST_PER_TASK = 0.015  # ~$0.015 per task with gemini-2.5-pro
        FAST_COST_PER_TASK = 0.002   # ~$0.002 per task with gemini-3.5-flash
        
        routing_cost = 0.0
        uniform_smart_cost = len(tasks) * SMART_COST_PER_TASK
        
        for task in tasks:
            model_key = self._routing_table.get(
                (task.get("review_type", "style"), task.get("priority", "low")),
                "fast"
            )
            routing_cost += SMART_COST_PER_TASK if model_key == "smart" else FAST_COST_PER_TASK
        
        savings_pct = int((1 - routing_cost / uniform_smart_cost) * 100)
        
        return {
            "uniform_smart_cost": f"${uniform_smart_cost:.4f}",
            "routing_cost": f"${routing_cost:.4f}",
            "savings_pct": f"{savings_pct}%",
            "savings_usd": f"${uniform_smart_cost - routing_cost:.4f}"
        }
