import unittest

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import torch


class TestEMAUtils(unittest.TestCase):
    def test_update_ema_state_matches_formula(self) -> None:
        from train_gpt import update_ema_state_

        ema = {"w": torch.tensor([1.0, 2.0], dtype=torch.float32)}
        model = {"w": torch.tensor([3.0, 6.0], dtype=torch.float32)}
        update_ema_state_(ema, model, decay=0.5)
        self.assertTrue(torch.allclose(ema["w"], torch.tensor([2.0, 4.0], dtype=torch.float32)))

        # Second update should apply on the updated EMA value.
        model2 = {"w": torch.tensor([7.0, 7.0], dtype=torch.float32)}
        update_ema_state_(ema, model2, decay=0.5)
        # ema <- 0.5*[2,4] + 0.5*[7,7] = [4.5, 5.5]
        self.assertTrue(torch.allclose(ema["w"], torch.tensor([4.5, 5.5], dtype=torch.float32)))


if __name__ == "__main__":
    unittest.main()

