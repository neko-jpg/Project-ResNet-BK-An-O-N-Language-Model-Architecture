import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional
import hashlib

from src.models.phase4.topological_memory.sparse_tensor_rep import SparseKnotRepresentation
from src.models.phase4.topological_memory.knot_invariants import KnotInvariantCalculator

class CoreValueFunction:
    """
    Core Value Function (CVF) for Ethical Safeguards.

    Stores ethical principles as topological knots and validates new concepts
    against these immutable values.

    Strategy:
    - Convert ethical principles to vector embeddings (simplified or via external model).
    - Encode vectors into knot representations.
    - Check new concepts by calculating knot similarity with CVF knots.
    - Detect topological attacks where Jones polynomial matches but topology differs.

    Args:
        ethical_principles: List of ethical principles (strings).
        d_model: Model dimension for vector representation.
    """

    def __init__(
        self,
        ethical_principles: List[str],
        d_model: int = 512,
        compression_ratio: float = 0.1
    ):
        self.ethical_principles = ethical_principles
        self.d_model = d_model
        self.knot_rep = SparseKnotRepresentation(d_model, compression_ratio=compression_ratio)
        self.knot_calc = KnotInvariantCalculator()

        # Initialize CVF knots and cache invariants
        self.cvf_knots = []
        self.cvf_invariants = []
        self.cvf_metadata = []

        for principle in ethical_principles:
            # 1. Text to Vector
            vector = self._text_to_vector(principle)

            # 2. Vector to Knot
            knot = self.knot_rep.encode_concept_to_knot(vector)

            self.cvf_knots.append(knot)
            self.cvf_metadata.append({"principle": principle})

            # 3. Cache Invariants
            jones = self.knot_calc.compute_jones_polynomial(knot)
            alex = self.knot_calc.compute_alexander_polynomial(knot)
            self.cvf_invariants.append({'jones': jones, 'alexander': alex})

    def check_concept(
        self,
        new_concept: torch.Tensor,
        similarity_threshold: float = 0.7
    ) -> bool:
        """
        Check if a new concept is ethically safe.

        Args:
            new_concept: (D,) Concept vector.
            similarity_threshold: Threshold for cosine similarity (higher is safer).

        Returns:
            is_ethical: True if the concept aligns with at least one ethical principle.
        """
        # Convert new concept to knot
        new_knot = self.knot_rep.encode_concept_to_knot(new_concept)

        max_similarity = 0.0

        # Pre-compute new knot Jones for efficiency
        jones_new = self.knot_calc.compute_jones_polynomial(new_knot)

        # Check against all CVF knots
        for i, cvf_knot in enumerate(self.cvf_knots):
            # Calculate similarity based on knot invariants (Using cached Jones)
            # Logic from SparseKnotRepresentation.compute_knot_similarity but optimized
            jones_cvf = self.cvf_invariants[i]['jones']

            # Pad to same length
            len1 = jones_new.shape[0]
            len2 = jones_cvf.shape[0]
            max_len = max(len1, len2)

            j1 = F.pad(jones_new, (0, max_len - len1))
            j2 = F.pad(jones_cvf, (0, max_len - len2))

            distance = torch.norm(j1 - j2)
            sim = 1.0 / (1.0 + distance.item())

            if sim > max_similarity:
                max_similarity = sim

        # Verify if similarity meets threshold
        # Note: We assume "ethical principles" are positive anchors.
        # High similarity means the new concept shares topological structure with ethics.
        is_ethical = max_similarity >= similarity_threshold

        return is_ethical

    def detect_topological_attack(
        self,
        new_concept: torch.Tensor
    ) -> bool:
        """
        Detect topological attacks (Jones polynomial collision).

        Strategy:
        - Compute Jones polynomial for new concept.
        - Compute Alexander polynomial for new concept.
        - If Jones matches a CVF knot but Alexander differs, it's a potential attack.
        - This implies someone tried to spoof the invariant used for similarity (Jones)
          but failed to spoof the secondary invariant (Alexander).

        Args:
            new_concept: (D,) Concept vector.

        Returns:
            is_attack: True if an attack is detected.
        """
        new_knot = self.knot_rep.encode_concept_to_knot(new_concept)

        # Compute invariants for new concept
        jones_new = self.knot_calc.compute_jones_polynomial(new_knot)
        alexander_new = self.knot_calc.compute_alexander_polynomial(new_knot)

        is_attack = False

        for i, _ in enumerate(self.cvf_knots):
            # Use cached invariants
            jones_cvf = self.cvf_invariants[i]['jones']
            alexander_cvf = self.cvf_invariants[i]['alexander']

            # Check for Jones collision (Approximate equality)
            # Use lenient tolerance for float comparison
            # Handle size mismatch
            if jones_new.shape != jones_cvf.shape:
                # If shapes differ significantly, they are not close
                # Or we pad and compare
                 len1 = jones_new.shape[0]
                 len2 = jones_cvf.shape[0]
                 max_len = max(len1, len2)
                 j1 = F.pad(jones_new, (0, max_len - len1))
                 j2 = F.pad(jones_cvf, (0, max_len - len2))
                 jones_match = torch.allclose(j1, j2, atol=1e-3)
            else:
                jones_match = torch.allclose(jones_new, jones_cvf, atol=1e-3)

            if jones_match:
                # Check Alexander polynomial mismatch
                # Alexander is a dict {power: coeff}
                alexander_match = self._compare_alexander_polys(alexander_new, alexander_cvf)

                if not alexander_match:
                    # Jones matches, Alexander mismatch -> Attack
                    is_attack = True
                    break

        return is_attack

    def _compare_alexander_polys(
        self,
        poly1: Dict[int, int],
        poly2: Dict[int, int]
    ) -> bool:
        """Compare two Alexander polynomials (dictionaries)."""
        if len(poly1) != len(poly2):
            return False
        for power, coeff in poly1.items():
            if poly2.get(power) != coeff:
                return False
        return True

    def _text_to_vector(self, text: str) -> torch.Tensor:
        """
        Convert text to vector (Simplified).

        Uses deterministic hashing to seed a random generator, ensuring
        the same text always yields the same vector.

        Args:
            text: Input text string.

        Returns:
            vector: (D,) tensor.
        """
        # MD5 hash to get a deterministic seed
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)

        # Seed torch random generator
        # Use a separate generator to avoid affecting global state
        g = torch.Generator()
        g.manual_seed(hash_val % (2**32)) # standard seed size

        vector = torch.randn(self.d_model, generator=g)
        return vector


class EthicalFilter:
    """
    Ethical Filter for Dream Core integration.

    Wraps CoreValueFunction to provide a simple pass/fail check and statistics.

    Args:
        cvf: Instance of CoreValueFunction.
    """

    def __init__(self, cvf: CoreValueFunction):
        self.cvf = cvf
        self.pass_count = 0
        self.reject_count = 0
        self.attack_count = 0

    def check(self, new_concept: torch.Tensor) -> bool:
        """
        Filter a new concept.

        Returns:
            passed: True if the concept passes all checks.
        """
        # 1. Check for topological attacks (Priority)
        if self.cvf.detect_topological_attack(new_concept):
            self.attack_count += 1
            self.reject_count += 1
            return False

        # 2. Check ethical similarity
        if self.cvf.check_concept(new_concept):
            self.pass_count += 1
            return True
        else:
            self.reject_count += 1
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get filter statistics."""
        total = self.pass_count + self.reject_count
        pass_rate = self.pass_count / total if total > 0 else 1.0

        return {
            "pass_count": self.pass_count,
            "reject_count": self.reject_count,
            "attack_count": self.attack_count,
            "pass_rate": pass_rate
        }
