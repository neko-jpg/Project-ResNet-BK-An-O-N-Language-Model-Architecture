
import torch
import unittest
from src.kernels.bk_scan import scan_op_impl

class TestBKScanLogic(unittest.TestCase):
    """
    Tests the logic of the Triton scan operation using a Python simulation.
    Verifies that the matrix multiplication order in scan_op correctly corresponds
    to the expected prefix product M_k ... M_1.
    """

    def test_scan_op_matrix_order(self):
        """
        Verify scan_op(Acc, Val) implements Val @ Acc (or Acc @ Val depending on definition).
        """

        # Define 2x2 complex matrices M1, M2
        # Random complex values
        torch.manual_seed(42)

        def random_complex_2x2():
            real = torch.randn(2, 2)
            imag = torch.randn(2, 2)
            return real, imag

        # Matrix A (Accumulator)
        a_r, a_i = random_complex_2x2()
        # Matrix B (Value)
        b_r, b_i = random_complex_2x2()

        # Flatten components for scan_op
        # a11r, a11i, a12r, a12i, a21r, a21i, a22r, a22i
        args_a = (
            a_r[0,0], a_i[0,0], a_r[0,1], a_i[0,1],
            a_r[1,0], a_i[1,0], a_r[1,1], a_i[1,1]
        )
        args_b = (
            b_r[0,0], b_i[0,0], b_r[0,1], b_i[0,1],
            b_r[1,0], b_i[1,0], b_r[1,1], b_i[1,1]
        )

        # Run scan_op_impl (python implementation)
        res_tuple = scan_op_impl(*args_a, *args_b)

        c_r = torch.tensor([[res_tuple[0], res_tuple[2]], [res_tuple[4], res_tuple[6]]])
        c_i = torch.tensor([[res_tuple[1], res_tuple[3]], [res_tuple[5], res_tuple[7]]])
        C_scan = torch.complex(c_r, c_i)

        # Reconstruct complex tensors for A and B
        A = torch.complex(a_r, a_i)
        B = torch.complex(b_r, b_i)

        # Calculate expected result: B @ A
        # Because in a prefix sum: Current = New_Value + Previous_Sum
        # Here: Current = New_Matrix @ Previous_Matrix
        # So op(Prev, New) should be New @ Prev

        Expected = torch.matmul(B, A)

        # Check correctness
        diff = (C_scan - Expected).abs().max().item()
        print(f"Max difference between scan_op and B@A: {diff}")

        self.assertTrue(diff < 1e-5, f"scan_op logic mismatch. Diff: {diff}")

    def test_full_sequence_logic(self):
        """
        Simulate a full scan over a sequence of 4 matrices to ensure
        M4 @ M3 @ M2 @ M1 order is preserved.
        """
        N = 4
        matrices = []
        for i in range(N):
            r, im = torch.randn(2, 2), torch.randn(2, 2)
            matrices.append(torch.complex(r, im))

        # Calculate expected prefix products using loop
        # Order: M1, M2@M1, M3@M2@M1 ...
        expected_prefixes = []
        current = matrices[0]
        expected_prefixes.append(current)

        for i in range(1, N):
            # P_i = M_i @ P_{i-1}
            current = torch.matmul(matrices[i], current)
            expected_prefixes.append(current)

        # Simulate Triton scan
        scan_result = [matrices[0]]

        # Flatten logic to use scan_op_impl
        def to_flat(m):
            r, i = m.real, m.imag
            return (
                r[0,0], i[0,0], r[0,1], i[0,1],
                r[1,0], i[1,0], r[1,1], i[1,1]
            )

        def from_flat(args):
            r = torch.tensor([[args[0], args[2]], [args[4], args[6]]])
            i = torch.tensor([[args[1], args[3]], [args[5], args[7]]])
            return torch.complex(r, i)

        accum = matrices[0]

        for i in range(1, N):
            val = matrices[i]

            acc_flat = to_flat(accum)
            val_flat = to_flat(val)

            res_flat = scan_op_impl(*acc_flat, *val_flat)
            accum = from_flat(res_flat)

            scan_result.append(accum)

        # Compare
        for i in range(N):
            diff = (scan_result[i] - expected_prefixes[i]).abs().max().item()
            self.assertTrue(diff < 1e-5, f"Sequence mismatch at index {i}")

if __name__ == '__main__':
    unittest.main()
