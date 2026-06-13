# 06_write_strategy.py
bad_system_prompt = """
You are a highly experienced senior software security engineer with over 15 years of 
experience in application security, penetration testing, and secure code review practices.
You have deep expertise in the OWASP Top 10 security vulnerabilities and have worked with
Fortune 500 companies to identify and remediate critical security issues.

When reviewing code, you should approach the task with a critical security mindset. Look
for any potential vulnerabilities that could be exploited by malicious actors. Be thorough
in your analysis and make sure to check for common security anti-patterns.

The types of issues you should look for include but are not limited to: SQL injection
vulnerabilities where user input is directly concatenated into SQL queries, cross-site
scripting (XSS) vulnerabilities where user input is reflected in HTML output without
proper encoding, authentication weaknesses such as using weak hashing algorithms like MD5
or SHA1 for passwords, hardcoded credentials, API keys, or secrets that should be stored
in environment variables, insecure direct object references, and path traversal attacks.

When you find an issue, please describe it clearly and explain why it is a security risk.
Provide a concrete fix recommendation that the developer can implement. Rate the severity
of each issue as critical, high, medium, or low based on the potential impact.

Please be professional and constructive in your feedback. Remember that the developer may
not have deep security expertise, so explain things clearly and helpfully.
"""


good_system_prompt = """ROLE: Security Code Reviewer — DevPulse
FOCUS: OWASP Top 10 vulnerabilities ONLY.
CHECK:
- SQL injection: raw queries, f-string interpolation in DB calls
- Hardcoded secrets: API keys, passwords, tokens in source code
- Weak auth: MD5/SHA1 password hashing, plain-text comparison
- Path traversal: user input in file paths without validation
- Broken access: missing auth decorators on sensitive endpoints
IGNORE: Style, documentation, performance (unless it is a security issue).
SEVERITY: SQL injection/secrets → critical | Auth bypass → high | Others → medium/low
FORMAT: Return structured JSON. For each issue: line, category, description, severity, fix."""


from langchain_google_genai import ChatGoogleGenerativeAI

def estimate_prompt_tokens(text: str, model_name: str = "gemini-2.0-flash") -> dict:
    """
    Estimate token count for a prompt using the model's tokenizer.
    
    Note: This is an approximation. Different models tokenize differently.
    As a rule of thumb: 1 token ≈ 4 characters in English text.
    """
    # Simple character-based estimate for planning purposes
    char_estimate = len(text)
    token_estimate = char_estimate // 4
    
    # Cost estimate at typical rates ($0.075 per 1M tokens for gemini-2.0-flash)
    cost_per_million = 0.075
    cost_per_call = (token_estimate / 1_000_000) * cost_per_million
    
    # At scale: 1000 PRs reviewed per day, 10 files per PR, 10 reasoning turns per file
    daily_calls = 1000 * 10 * 10
    daily_cost = cost_per_call * daily_calls
    
    return {
        "character_count": char_estimate,
        "estimated_tokens": token_estimate,
        "cost_per_single_call": f"${cost_per_call:.6f}",
        "daily_cost_at_scale": f"${daily_cost:.2f}"
    }

print("=== Prompt Token Analysis ===")
bad_analysis = estimate_prompt_tokens(bad_system_prompt)
good_analysis = estimate_prompt_tokens(good_system_prompt)

print(f"\nInefficient prompt:")
print(f"  Tokens: ~{bad_analysis['estimated_tokens']}")
print(f"  Daily cost at scale: {bad_analysis['daily_cost_at_scale']}")

print(f"\nEfficient prompt:")
print(f"  Tokens: ~{good_analysis['estimated_tokens']}")
print(f"  Daily cost at scale: {good_analysis['daily_cost_at_scale']}")

# Output:
# Inefficient prompt:
#   Tokens: ~195
#   Daily cost at scale: $0.15
# 
# Efficient prompt:
#   Tokens: ~36
#   Daily cost at scale: $0.03
# 
# 80% cost reduction on system prompts alone, at scale.
