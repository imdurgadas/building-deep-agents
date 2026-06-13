# Production Deep Agents - Companion Code

This repository contains the complete, runnable companion code files for the **Domain-Specific Deep Agents** blog series — a deep dive into production-grade cognitive architectures, planning modules, self-correcting memory, subagent parallelism, and observability.

---

## Repository Structure

To maintain compatibility with both python import rules and the blog post text, the code is structured as follows:
- **Numbered files** (e.g., `01_planner.py`): Keep original filenames as referenced in the blog posts for easy matching.
- **Clean-named copies** (e.g., `planner.py`): Provide clean python modules to resolve python's syntax limitations (Python throws a `SyntaxError` when importing from module names starting with a digit). All internal imports in these scripts have been refactored to use the clean module names.

```text
building-deep-agents/
├── requirements.txt            # Project dependencies
├── README.md                   # Setup and usage guide
├── 01_workspace.py             # Durable state workspace & filesystem
├── 01_planner.py               # Deep agent planning phase
├── 02_harness_setup.py         # Resilient LLM config and rate limits
├── 03_github_tools.py          # Git & mock API integrations
├── 04_child_agent.py           # Isolated single-file reviewer nodes
├── 05_parallel_executor.py     # Thread pool child execution
├── 06_write_strategy.py        # Token compression and system prompt cost management
├── 07_select_strategy.py       # AST parsing code locator
├── 08_compress_strategy.py     # Context manager, workspace offloader & orchestrator
├── 09_production_checkpointing.py # Durable state graph checkpointers
├── 10_langsmith_tracing.py     # Production telemetry and metadata tracking
├── 11_hitl_approval.py         # Human-in-the-loop validation
├── 12_cost_safeguards.py       # Budget limits, token gates & auto-abort
├── 12_custom_harness.py        # Integrated domain-specific harness
├── 13_model_router.py          # Dynamic LLM routing (flash vs. reasoning)
├── 14_tool_registry.py         # Semantic tool search & authorization registry
├── src/utils.py                # Local helper utilities
└── [clean_copies].py           # Clean module names (e.g. model_router.py)
```

---

## Prerequisites & Installation

1. **Navigate** to the project directory:
   ```bash
   cd building-deep-agents
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install the dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## Environment Configuration

Create a `.env` file in the root of this directory:

```env
# Google AI Studio API Key (for Gemini models)
GOOGLE_API_KEY=your_gemini_api_key_here

# LangSmith Tracing (Optional - for Part 5/Part 10 tracing)
LANGSMITH_API_KEY=your_langsmith_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=DevPulse-Production

# GitHub Access Token (Optional - for GitHub mock/real tools)
GITHUB_TOKEN=your_github_token_here
```

---

## How to Run the Scripts

Always ensure your virtual environment is active and environment variables are loaded. You can run any script directly using `python`:

### Example: Running Deep Agent Custom Harness
```bash
python custom_harness.py
```

*Note: You can run both the numbered files (e.g., `python 12_custom_harness.py`) or the clean-named files (e.g., `python custom_harness.py`). Both versions work identically and use the updated import structure.*
