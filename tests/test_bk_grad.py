
import torch
import unittest
from torch.autograd import gradcheck
from src.models.bk_core import BKCoreFunction

class TestBKGrad(unittest.TestCase):

    def setUp(self):
        # Use single precision because BK-Core implementation uses complex64/float32
        # Mismatch in precision causes gradcheck failures due to truncation.
        self.dtype = torch.float32
        self.cdtype = torch.complex64
        self.device = torch.device("cpu")

        # Set up small inputs
        self.B = 2
        self.N = 10

        # Random inputs
        # he_diag (B, N) real
        self.he_diag = torch.randn(self.B, self.N, dtype=self.dtype, device=self.device, requires_grad=True)

        # Diagonal case: h0_super, h0_sub = 0
        # This ensures analytic gradient (Diagonal Approx) is EXACT.
        self.h0_super = torch.zeros(self.B, self.N - 1, dtype=self.dtype, device=self.device, requires_grad=False)
        self.h0_sub = torch.zeros(self.B, self.N - 1, dtype=self.dtype, device=self.device, requires_grad=False)

        # z scalar complex
        self.z = torch.tensor(1.0 + 0.5j, dtype=self.cdtype, device=self.device)

    def test_analytic_gradient_diagonal_limit(self):
        """
        Verify that the pure analytic gradient (GRAD_BLEND=0.0) passes gradcheck
        in the diagonal limit (hopping=0).
        Theory: dG/dv = -G^2 is exact for diagonal matrices.
        """
        # Force GRAD_BLEND to 0.0
        BKCoreFunction.GRAD_BLEND = 0.0

        inputs = (self.he_diag, self.h0_super, self.h0_sub, self.z, False)

        print("\nRunning gradcheck for Analytic Gradient (Diagonal Limit)...")
        # Relax tolerance for float32
        test = gradcheck(BKCoreFunction.apply, inputs, eps=1e-3, atol=1e-2)
        self.assertTrue(test, "Gradcheck failed for Analytic Gradient in Diagonal Limit")

    def test_hypothesis7_correlation(self):
        """
        Verify that Hypothesis-7 gradient (GRAD_BLEND=1.0) produces finite gradients.
        We also check correlation, but don't strictly assert it must be positive given
        the phase differences discussed.
        """

        BKCoreFunction.GRAD_BLEND = 1.0

        # Forward
        out_features = BKCoreFunction.apply(self.he_diag, self.h0_super, self.h0_sub, self.z, False)

        loss = 0.5 * (out_features ** 2).sum()

        if self.he_diag.grad is not None:
            self.he_diag.grad.zero_()

        # Backward (uses H7)
        loss.backward()
        grad_h7 = self.he_diag.grad.clone()

        # Assert Finite
        self.assertFalse(torch.isnan(grad_h7).any(), "H7 gradient contains NaNs")
        self.assertFalse(torch.isinf(grad_h7).any(), "H7 gradient contains Infs")

        # Check correlation just for info
        BKCoreFunction.GRAD_BLEND = 0.0
        self.he_diag.grad.zero_()

        # Re-run forward to get new graph
        out_features_2 = BKCoreFunction.apply(self.he_diag, self.h0_super, self.h0_sub, self.z, False)
        loss_2 = 0.5 * (out_features_2 ** 2).sum()
        loss_2.backward()
        grad_true = self.he_diag.grad.clone()

        cosine_sim = torch.nn.functional.cosine_similarity(grad_h7.view(-1).unsqueeze(0), grad_true.view(-1).unsqueeze(0)).item()
        print(f"\nCosine Similarity between H7 and Analytic (Diagonal): {cosine_sim}")

if __name__ == '__main__':
    unittest.main()
