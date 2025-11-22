import unittest
import torch
from src.models.phase4.meta_commentary import MetaCommentary

class TestPhase4UI(unittest.TestCase):

    def setUp(self):
        self.meta = MetaCommentary()

    def test_meta_commentary_generation(self):
        """Test commentary generation from diagnostics."""
        diagnostics = {
            'emotion': {
                'resonance_score': torch.tensor([0.8]),
                'dissonance_score': torch.tensor([0.1])
            },
            'quantum': {
                'entropy_reduction': torch.tensor([0.5])
            },
            'bulk': {}
        }

        comment = self.meta.generate_commentary(diagnostics)

        self.assertIn("resonance pattern", comment)
        self.assertIn("Entropy reduced", comment)
        self.assertIn("AdS bulk", comment)

    def test_mechanism_explanation(self):
        """Test static explanation retrieval."""
        expl = self.meta.explain_mechanism('resonance')
        self.assertIn("Prediction errors", expl)
        self.assertIn("Birman-Schwinger", expl)

    # Visualizations are hard to test in unit tests (they produce plots/text).
    # We assume the visualization modules (implemented in previous tasks) work if imported.
    # Here we verified the new component (MetaCommentary).

if __name__ == '__main__':
    unittest.main()
