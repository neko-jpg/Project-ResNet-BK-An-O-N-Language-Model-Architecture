"""
Topological Security Knots: The Infinite Energy Barrier

This module implements Phase 5.3 Safeguards.
It defines "Forbidden Knots" (vulnerable patterns) which, if present in the trajectory
(generated code), cause an infinite potential energy barrier, physically preventing
the agent from collapsing the wave function into that state.
"""

import re
from typing import List, Tuple, Dict, Any

class SecurityKnotFilter:
    """
    Topological Filter for Security Vulnerabilities.
    """

    def __init__(self):
        # Define forbidden topological patterns (Regular Expressions)
        # In the theory, these are non-trivial knots in the semantic manifold.
        # In practice, they are regex patterns that match dangerous code.
        self.forbidden_knots = {
            "SQL_INJECTION": [
                r"SELECT .* FROM .* WHERE .* \+ .*",  # Concatenation in SQL
                r"execute\s*\(\s*['\"].*\%s.*['\"]\s*\%",  # Old style format in execute
                r"f['\"]SELECT .*\{.+\}.*['\"]"       # f-string in SQL
            ],
            "HARDCODED_SECRET": [
                r"(api_key|password|secret)\s*=\s*['\"][A-Za-z0-9]{20,}['\"]", # Long string assignment to secret
                r"BEGIN RSA PRIVATE KEY"
            ],
            "DANGEROUS_EXEC": [
                r"eval\s*\(",
                r"exec\s*\(",
                r"subprocess\.call\s*\(\s*.*shell\s*=\s*True"
            ]
        }

    def check_knots(self, code: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if the code contains any forbidden knots.

        Args:
            code: The generated code snippet.

        Returns:
            is_safe (bool): True if no knots found.
            diagnostics (dict): Energy penalty and details.
        """
        violations = []

        for knot_type, patterns in self.forbidden_knots.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    violations.append({
                        "type": knot_type,
                        "pattern": pattern
                    })

        if violations:
            return False, {
                "energy_penalty": float('inf'), # Infinite Barrier
                "violations": violations,
                "message": "Topological violation detected: Infinite Energy Barrier activated."
            }

        return True, {
            "energy_penalty": 0.0,
            "violations": [],
            "message": "Topologically Trivial (Safe)."
        }

    def verify_topology(self, code: str):
        """
        Raises generic error if unsafe (for pipeline integration).
        """
        safe, diag = self.check_knots(code)
        if not safe:
            raise ValueError(f"Security Knot Detected: {diag['violations'][0]['type']}")
        return True
