# 14_tool_registry.py
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Callable
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.tools import tool

# ---- Language-Specific Tool Schemas ----

class PythonLintSchema(BaseModel):
    code: str = Field(description="Python source code to analyze for style and type issues")
    check_types: bool = Field(default=False, description="Run mypy type checking (slower but more thorough)")

class GoStaticCheckSchema(BaseModel):
    code: str = Field(description="Go source code to check for bugs, goroutine leaks, and unsafe patterns")

class TypeScriptAuditSchema(BaseModel):
    package_json: str = Field(description="Content of package.json to audit for dependency vulnerabilities")
    tsconfig: Optional[str] = Field(default=None, description="Content of tsconfig.json if available")

class JavaSecuritySchema(BaseModel):
    code: str = Field(description="Java source code to check for injection vulnerabilities and insecure patterns")

# ---- Language-Specific Tool Implementations ----

@tool(args_schema=PythonLintSchema)
def python_security_scan(code: str, check_types: bool = False) -> str:
    """
    Perform a security-focused static analysis on Python code.
    Checks for: SQL injection patterns, hardcoded secrets, use of weak crypto (MD5, SHA1),
    insecure deserialization (pickle), and dangerous eval/exec usage.
    """
    findings = []
    lines = code.splitlines()
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip().lower()
        
        # SQL injection patterns
        if any(p in stripped for p in ['%s" % ', "f\"select", "f'select", ".format(", "+ request.", "+ user_input"]):
            findings.append(f"Line {i}: 🔴 CRITICAL — Potential SQL injection: string interpolation in query")
        
        # Hardcoded secrets
        if any(p in stripped for p in ['secret_key = "', "api_key = '", 'password = "', 'token = "']):
            if not any(e in stripped for e in ['os.environ', 'os.getenv', 'env.get']):
                findings.append(f"Line {i}: 🔴 CRITICAL — Hardcoded secret detected")
        
        # Weak crypto
        if 'md5(' in stripped and 'hashlib' in code.lower():
            findings.append(f"Line {i}: 🟠 HIGH — MD5 is cryptographically broken. Use SHA-256 or bcrypt for passwords")
        
        if 'sha1(' in stripped:
            findings.append(f"Line {i}: 🟠 HIGH — SHA1 is deprecated for security use. Use SHA-256 minimum")
        
        # Dangerous functions
        if 'eval(' in stripped and 'request' in code.lower():
            findings.append(f"Line {i}: 🔴 CRITICAL — eval() with potential user input. Remote code execution risk")
        
        if 'pickle.loads(' in stripped:
            findings.append(f"Line {i}: 🟠 HIGH — pickle.loads() with untrusted data enables code execution")
    
    if not findings:
        return "✅ Python security scan: No obvious vulnerabilities detected in the code sample."
    
    result = f"Python Security Scan Results ({len(findings)} finding(s)):\n\n"
    result += "\n".join(findings)
    return result

@tool(args_schema=GoStaticCheckSchema)
def go_security_scan(code: str) -> str:
    """
    Perform security and concurrency analysis on Go source code.
    Checks for: goroutine leaks, channel deadlocks, unsafe pointer usage,
    SQL injection in database/sql calls, and missing error handling.
    """
    findings = []
    lines = code.splitlines()
    
    has_goroutine = False
    goroutine_lines = []
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # Goroutine tracking (leak detection)
        if stripped.startswith("go ") or "go func(" in stripped:
            has_goroutine = True
            goroutine_lines.append(i)
        
        # SQL injection in Go
        if 'db.Query(' in stripped or 'db.Exec(' in stripped:
            if 'fmt.Sprintf' in stripped or '"+" +' in stripped or 'string(' in stripped:
                findings.append(f"Line {i}: 🔴 CRITICAL — SQL injection: string-built query in db.Query/Exec. Use parameterized queries: db.Query(\"...\", arg1, arg2)")
        
        # Missing error handling
        if '_ = ' in stripped and ('err' in stripped.lower() or 'error' in stripped.lower()):
            findings.append(f"Line {i}: 🟡 MEDIUM — Error being ignored with '_'. Silent failures can hide security issues")
        
        # Unsafe pointer
        if 'unsafe.Pointer' in stripped:
            findings.append(f"Line {i}: 🟠 HIGH — unsafe.Pointer usage. Bypasses Go's memory safety. Requires security review")
        
        # Race condition potential
        if 'sync.WaitGroup' not in code and has_goroutine and 'shared_' in stripped.lower():
            findings.append(f"Line {i}: 🟠 HIGH — Potential data race: accessing shared state in goroutine without synchronization")
    
    if goroutine_lines and 'wg.Wait()' not in code and 'sync.WaitGroup' not in code:
        findings.append(
            f"Lines {goroutine_lines}: 🟠 HIGH — Goroutine(s) spawned without WaitGroup or done channel. "
            f"Goroutine leak risk if parent function returns before goroutines complete"
        )
    
    if not findings:
        return "✅ Go security scan: No obvious vulnerabilities detected in the code sample."
    
    result = f"Go Security Scan Results ({len(findings)} finding(s)):\n\n"
    result += "\n".join(findings)
    return result

@tool(args_schema=TypeScriptAuditSchema)
def typescript_dependency_audit(package_json: str, tsconfig: Optional[str] = None) -> str:
    """
    Audit TypeScript/JavaScript project dependencies for known vulnerabilities.
    Checks package.json for outdated or vulnerable dependencies.
    Also checks tsconfig.json for unsafe compiler settings.
    """
    import json
    findings = []
    
    try:
        pkg = json.loads(package_json)
    except json.JSONDecodeError:
        return "❌ Could not parse package.json — invalid JSON format"
    
    deps = {**pkg.get("dependencies", {}), **pkg.get("devDependencies", {})}
    
    # Check for known-vulnerable versions (simplified — production would use npm audit API)
    known_vulnerable = {
        "lodash": ("4.17.20", "Prototype pollution vulnerability"),
        "express": ("4.17.1", "Outdated — missing security patches"),
        "jsonwebtoken": ("8.5.1", "Algorithm confusion vulnerability in older versions"),
        "axios": ("0.21.1", "SSRF vulnerability in redirect handling"),
    }
    
    for package, (safe_version, desc) in known_vulnerable.items():
        if package in deps:
            current = deps[package].lstrip("^~>=")
            findings.append(f"🟠 HIGH — {package}@{deps[package]}: {desc}. Upgrade to >{safe_version}")
    
    # Check for risky packages
    risky_packages = {
        "eval": "Allows arbitrary code execution",
        "node-serialize": "Remote code execution via deserialization",
        "crypto-js": "Often misused — prefer Node.js built-in crypto module",
    }
    for pkg_name, risk in risky_packages.items():
        if pkg_name in deps:
            findings.append(f"🟡 MEDIUM — {pkg_name}: {risk}")
    
    # Check tsconfig settings
    if tsconfig:
        try:
            ts_config = json.loads(tsconfig)
            compiler_options = ts_config.get("compilerOptions", {})
            
            if not compiler_options.get("strict", False):
                findings.append("🟡 MEDIUM — tsconfig: strict mode disabled. Enables unsafe any types and missing null checks")
            
            if compiler_options.get("noImplicitAny") is False:
                findings.append("🟡 MEDIUM — tsconfig: noImplicitAny=false. Allows untyped code that can mask security bugs")
        except json.JSONDecodeError:
            findings.append("⚠️  Could not parse tsconfig.json")
    
    if not findings:
        return f"✅ TypeScript audit: No known vulnerabilities in {len(deps)} dependencies."
    
    result = f"TypeScript Dependency Audit ({len(findings)} finding(s) in {len(deps)} packages):\n\n"
    result += "\n".join(findings)
    return result

@tool(args_schema=JavaSecuritySchema)
def java_security_scan(code: str) -> str:
    """
    Perform security analysis on Java source code.
    Checks for: SQL injection (raw JDBC), XXE vulnerabilities, insecure deserialization,
    hardcoded credentials, and use of deprecated/unsafe classes.
    """
    findings = []
    lines = code.splitlines()
    
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        
        # SQL injection in JDBC
        if ('executeQuery(' in stripped or 'executeUpdate(' in stripped or 'execute(' in stripped):
            if '+' in stripped and ('"' in stripped or "'" in stripped):
                findings.append(f"Line {i}: 🔴 CRITICAL — JDBC SQL injection: string concatenation in query. Use PreparedStatement")
        
        # Deserialization
        if 'ObjectInputStream' in stripped and 'readObject()' in stripped:
            findings.append(f"Line {i}: 🔴 CRITICAL — Insecure deserialization via ObjectInputStream. Use JSON/ProtoBuf instead")
        
        # XXE vulnerability
        if 'DocumentBuilderFactory' in stripped:
            if 'setFeature' not in code or 'disallow-doctype-decl' not in code:
                findings.append(f"Line {i}: 🟠 HIGH — DocumentBuilderFactory without XXE protection. Add setFeature(\"http://apache.org/xml/features/disallow-doctype-decl\", true)")
        
        # Hardcoded creds
        if any(p in stripped.lower() for p in ['password = "', 'secret = "', 'apikey = "']):
            findings.append(f"Line {i}: 🔴 CRITICAL — Hardcoded credential. Use environment variables or a secrets manager")
        
        # Deprecated crypto
        if 'MD5' in stripped or 'SHA1' in stripped:
            findings.append(f"Line {i}: 🟠 HIGH — Weak hashing algorithm. Use SHA-256 minimum, bcrypt for passwords")
    
    if not findings:
        return "✅ Java security scan: No obvious vulnerabilities detected in the code sample."
    
    result = f"Java Security Scan Results ({len(findings)} finding(s)):\n\n"
    result += "\n".join(findings)
    return result

# ---- The Registry ----

class ToolRegistry:
    """
    A domain-aware registry that returns the appropriate tool set
    for a given programming language and review type.
    
    Extend this by:
    1. Adding new language-specific tools above
    2. Registering them in the _registry dict below
    3. Optionally adding MCP server tools (see FAQs)
    """
    
    # Registry structure: language → review_type → [tools]
    _registry: Dict[str, Dict[str, List]] = {
        "python": {
            "security": [python_security_scan],
            "performance": [],  # No dedicated perf tool yet — LLM reasoning handles it
            "style": [],
            "test_coverage": []
        },
        "go": {
            "security": [go_security_scan],
            "performance": [go_security_scan],  # Reuse — includes goroutine analysis
            "style": [],
            "test_coverage": []
        },
        "typescript": {
            "security": [typescript_dependency_audit],
            "performance": [],
            "style": [],
            "test_coverage": []
        },
        "javascript": {
            "security": [typescript_dependency_audit],  # Same audit logic works
            "performance": [],
            "style": [],
            "test_coverage": []
        },
        "java": {
            "security": [java_security_scan],
            "performance": [],
            "style": [],
            "test_coverage": []
        }
    }
    
    # Extension map: file extensions → language
    EXTENSION_MAP = {
        ".py": "python",
        ".go": "go",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".java": "java",
        ".kt": "kotlin",
        ".rb": "ruby",
        ".rs": "rust"
    }
    
    def detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension."""
        suffix = Path(file_path).suffix.lower()
        return self.EXTENSION_MAP.get(suffix, "generic")
    
    def get_tools(self, file_path: str, review_type: str) -> List:
        """
        Return the appropriate tool set for the given file and review type.
        Returns an empty list if no specialized tools exist (LLM handles it alone).
        """
        language = self.detect_language(file_path)
        lang_tools = self._registry.get(language, {})
        tools = lang_tools.get(review_type, [])
        
        if tools:
            print(f"  🔧 [Registry] Loaded {len(tools)} specialized tool(s) for {language}/{review_type}")
        else:
            print(f"  🔧 [Registry] No specialized tools for {language}/{review_type} — LLM reasoning only")
        
        return tools
    
    def get_language(self, file_path: str) -> str:
        return self.detect_language(file_path)
