
import random
from typing import List, Dict, Optional

class MockDocumentFetcher:
    """
    Boundary Core: Document Fetcher.

    Simulates the retrieval of external knowledge (Boundary Conditions)
    to be injected into the Bulk (Language Model).

    Offline Mode: Returns pre-loaded scientific texts.
    """

    def __init__(self):
        # Small knowledge base of "scientific truths"
        self.knowledge_base = [
            "Riemann Hypothesis: Non-trivial zeros of zeta function have real part 1/2.",
            "Birman-Schwinger Principle: Eigenvalues of H correspond to singular values of K.",
            "AdS/CFT: Hyperbolic geometry of bulk corresponds to conformal field theory on boundary.",
            "Knot Theory: Jones polynomial is a topological invariant of knots.",
            "Quantum Mechanics: Wave function collapse is non-unitary measurement process.",
            "Ethics: Autonomy, Justice, Beneficence, Non-maleficence are core principles.",
            "Resonance: Constructive interference maximizes spectral density.",
            "Holography: Information in volume is encoded on surface area."
        ]

    def fetch(self, query: str = "", k: int = 1) -> List[str]:
        """
        Fetch documents relevant to query.
        (Mock: Returns random documents if query is simple, or keyword match).
        """
        # Simple keyword matching
        relevant = []
        if query:
            query_lower = query.lower()
            for doc in self.knowledge_base:
                if any(word in doc.lower() for word in query_lower.split()):
                    relevant.append(doc)

        # If no match, return random (Serendipity)
        if not relevant:
            return random.sample(self.knowledge_base, min(k, len(self.knowledge_base)))

        # Return top k relevant
        return relevant[:k]
