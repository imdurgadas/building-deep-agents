# 07_select_strategy.py
import ast
import os
from pathlib import Path
from typing import Dict, List, Optional
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# ---- Symbol Index Builder ----
# In production, this would be built incrementally as files change.
# For DevPulse, we build it once per PR review run.

class CodeSymbol(BaseModel):
    name: str
    kind: str  # 'function', 'class', 'method'
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None

class CodebaseIndex:
    """
    An index of code symbols (functions, classes) mapped to their file locations.
    Allows agents to load specific symbols without loading entire files.
    """
    
    def __init__(self):
        self._symbols: Dict[str, CodeSymbol] = {}
    
    def build_from_directory(self, directory: str) -> None:
        """Parse Python files in directory and index all symbols."""
        for py_file in Path(directory).rglob("*.py"):
            self._parse_file(str(py_file))
        
        print(f"📚 Codebase index built: {len(self._symbols)} symbols indexed")
    
    def _parse_file(self, file_path: str) -> None:
        """Extract functions and classes from a Python file using AST."""
        try:
            with open(file_path) as f:
                source = f.read()
            
            tree = ast.parse(source)
            lines = source.splitlines()
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    kind = "class" if isinstance(node, ast.ClassDef) else "function"
                    name = node.name
                    start_line = node.lineno
                    end_line = max(
                        getattr(child, "lineno", start_line)
                        for child in ast.walk(node)
                        if hasattr(child, "lineno")
                    )
                    
                    # Extract docstring if present
                    docstring = ast.get_docstring(node)
                    
                    self._symbols[name] = CodeSymbol(
                        name=name,
                        kind=kind,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        docstring=docstring
                    )
        except SyntaxError:
            pass  # Skip files with syntax errors
    
    def search(self, query: str) -> List[CodeSymbol]:
        """Search for symbols matching the query string."""
        query_lower = query.lower()
        return [
            symbol for name, symbol in self._symbols.items()
            if query_lower in name.lower()
        ]
    
    def get_symbol_source(self, symbol_name: str, source_root: str = ".") -> Optional[str]:
        """Load only the source lines for a specific symbol."""
        symbol = self._symbols.get(symbol_name)
        if not symbol:
            return None
        
        try:
            with open(symbol.file_path) as f:
                lines = f.readlines()
            
            return "".join(lines[symbol.start_line - 1:symbol.end_line])
        except (FileNotFoundError, IndexError):
            return f"# Source for {symbol_name} not available locally"

# Singleton index — built once per run, shared across all child agents
_codebase_index = CodebaseIndex()

# ---- LangChain Tools for Select Strategy ----

class SearchSymbolSchema(BaseModel):
    query: str = Field(description="Function or class name to search for in the codebase")

class LoadSymbolSchema(BaseModel):
    symbol_name: str = Field(description="Exact function or class name to load source code for")

@tool(args_schema=SearchSymbolSchema)
def search_codebase(query: str) -> str:
    """
    Search the codebase for functions or classes matching the query.
    Use this to discover what code exists before deciding what to load.
    Returns: list of matching symbol names, their file paths, and line numbers.
    """
    results = _codebase_index.search(query)
    
    if not results:
        return f"No symbols found matching '{query}' in the codebase index."
    
    output_lines = [f"Found {len(results)} symbol(s) matching '{query}':\n"]
    for symbol in results[:10]:  # Limit to 10 results
        output_lines.append(
            f"- {symbol.kind.upper()}: `{symbol.name}` in `{symbol.file_path}` "
            f"(lines {symbol.start_line}-{symbol.end_line})"
        )
        if symbol.docstring:
            output_lines.append(f"  Docstring: {symbol.docstring[:100]}")
    
    return "\n".join(output_lines)

@tool(args_schema=LoadSymbolSchema)
def load_symbol_source(symbol_name: str) -> str:
    """
    Load the complete source code for a specific function or class.
    Use this AFTER using search_codebase to confirm the symbol exists.
    Returns: The complete source code of the function or class.
    """
    # Try mock implementations first (for development)
    mock_sources = {
        "login_user": '''def login_user(request):
    username = request.POST.get('username', '')
    password = request.POST.get('password', '')
    # SQL injection vulnerability: f-string in query
    query = f"SELECT * FROM users WHERE username = '{username}'"
    user = db.execute(query).fetchone()
    if user:
        password_hash = md5(password).hexdigest()
        if password_hash == user.password_hash:
            return create_session(user)
    return None''',
        "create_token": '''def create_token(user_id: int) -> str:
    SECRET_KEY = os.environ.get("JWT_SECRET", "hardcoded-secret-123")
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + 3600
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")'''
    }
    
    if symbol_name in mock_sources:
        return mock_sources[symbol_name]
    
    # Try real filesystem lookup
    source = _codebase_index.get_symbol_source(symbol_name)
    if source:
        return source
    
    return f"Symbol '{symbol_name}' not found. Use search_codebase first to find the correct name."
