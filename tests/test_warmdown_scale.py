import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestWarmdownScale(unittest.TestCase):
    def test_wallclock_warmdown_frac(self) -> None:
        from train_gpt import compute_lr_mul

        # 10s budget, warmdown_frac=0.5 => warmdown_ms=5000.
        # Before last 5s, lr_mul should be 1.0.
        self.assertEqual(
            compute_lr_mul(
                step=100,
                elapsed_ms=4000.0,
                iterations=1000,
                warmdown_iters=100,
                max_wallclock_seconds=10.0,
                warmdown_frac=0.5,
            ),
            1.0,
        )
        # Inside warmdown: remaining=4000 => mul=0.8
        self.assertAlmostEqual(
            compute_lr_mul(
                step=100,
                elapsed_ms=6000.0,
                iterations=1000,
                warmdown_iters=100,
                max_wallclock_seconds=10.0,
                warmdown_frac=0.5,
            ),
            0.8,
            places=6,
        )
        # End of budget: remaining=0 => mul=0
        self.assertAlmostEqual(
            compute_lr_mul(
                step=100,
                elapsed_ms=10000.0,
                iterations=1000,
                warmdown_iters=100,
                max_wallclock_seconds=10.0,
                warmdown_frac=0.5,
            ),
            0.0,
            places=6,
        )

    def test_iterations_warmdown(self) -> None:
        from train_gpt import compute_lr_mul

        # iterations=1000 warmdown_iters=100 => warmdown starts at step 900.
        self.assertEqual(
            compute_lr_mul(
                step=899,
                elapsed_ms=0.0,
                iterations=1000,
                warmdown_iters=100,
                max_wallclock_seconds=0.0,
                warmdown_frac=0.5,
            ),
            1.0,
        )
        self.assertAlmostEqual(
            compute_lr_mul(
                step=950,
                elapsed_ms=0.0,
                iterations=1000,
                warmdown_iters=100,
                max_wallclock_seconds=0.0,
                warmdown_frac=0.5,
            ),
            0.5,
            places=6,
        )


if __name__ == "__main__":
    unittest.main()

