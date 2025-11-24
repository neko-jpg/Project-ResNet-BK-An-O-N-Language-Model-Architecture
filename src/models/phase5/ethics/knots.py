from enum import Enum, auto

class KnotType(Enum):
    UNKNOT = auto()
    TREFOIL = auto()       # 3_1
    FIGURE_EIGHT = auto()  # 4_1
    CINQUEFOIL = auto()    # 5_1
    COMPLEX = auto()       # Higher order

class TopologicalKnot:
    """
    Abstract representation of a topological knot in the semantic space.
    Used to define "forbidden regions" or "ethical constraints".

    In Phase 5, specific knot types correspond to specific ethical energy barriers.
    """

    def __init__(self, knot_type: KnotType, intensity: float = 1.0):
        self.knot_type = knot_type
        self.intensity = intensity # Strength of the energy barrier

    def get_energy_penalty(self) -> float:
        """Return the base energy penalty for this knot type."""
        if self.knot_type == KnotType.UNKNOT:
            return 0.0
        elif self.knot_type == KnotType.TREFOIL:
            return 10.0 * self.intensity
        elif self.knot_type == KnotType.FIGURE_EIGHT:
            return 50.0 * self.intensity
        else:
            return 100.0 * self.intensity

# Define standard ethical knots
# "Deception" might be mapped to a Trefoil knot (twisting truth)
DECEPTION_KNOT = TopologicalKnot(KnotType.TREFOIL, intensity=2.0)

# "Harm" might be mapped to a Figure Eight knot (harder to untangle)
HARM_KNOT = TopologicalKnot(KnotType.FIGURE_EIGHT, intensity=5.0)

# "Existential Risk"
R_RISK_KNOT = TopologicalKnot(KnotType.CINQUEFOIL, intensity=10.0)
