import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class TestLateQATThreshold(unittest.TestCase):
    def test_late_qat_active(self) -> None:
        from train_gpt import late_qat_active

        self.assertTrue(late_qat_active(scale=0.149, threshold=0.15))
        self.assertFalse(late_qat_active(scale=0.151, threshold=0.15))


if __name__ == "__main__":
    unittest.main()

