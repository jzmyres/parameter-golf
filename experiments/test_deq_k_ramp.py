import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import deq_maxk_ramp  # noqa: E402


class TestDeqKRamp(unittest.TestCase):
    def test_ramp_endpoints(self):
        # Ramp over 800 steps: step=1 -> 4, step=800 -> 12, step>800 -> 12
        self.assertEqual(deq_maxk_ramp(1, start=4, end=12, ramp_steps=800), 4)
        self.assertEqual(deq_maxk_ramp(800, start=4, end=12, ramp_steps=800), 12)
        self.assertEqual(deq_maxk_ramp(801, start=4, end=12, ramp_steps=800), 12)

    def test_monotone(self):
        ks = [deq_maxk_ramp(s, start=4, end=12, ramp_steps=50) for s in range(1, 51)]
        self.assertEqual(min(ks), 4)
        self.assertEqual(max(ks), 12)
        self.assertTrue(all(a <= b for a, b in zip(ks, ks[1:])))


if __name__ == "__main__":
    unittest.main()
