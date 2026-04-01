import os
import sys
import unittest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from train_gpt import _eval_seq_bounds


class TestFastValBounds(unittest.TestCase):
    def test_fast_val_partitions_fixed_prefix(self):
        total_seqs = 1000
        eval_batch_seqs = 32
        world_size = 8
        covered = 0
        last_end = 0
        for rank in range(world_size):
            start, end, global_eval = _eval_seq_bounds(
                total_seqs, eval_batch_seqs, rank, world_size, full_eval=False
            )
            self.assertEqual(global_eval, eval_batch_seqs)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, eval_batch_seqs)
            self.assertGreaterEqual(start, last_end)
            covered += end - start
            last_end = end
        self.assertEqual(covered, eval_batch_seqs)

    def test_full_val_partitions_entire_set(self):
        total_seqs = 1000
        eval_batch_seqs = 32
        world_size = 8
        covered = 0
        last_end = 0
        for rank in range(world_size):
            start, end, global_eval = _eval_seq_bounds(
                total_seqs, eval_batch_seqs, rank, world_size, full_eval=True
            )
            self.assertEqual(global_eval, total_seqs)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, total_seqs)
            self.assertGreaterEqual(start, last_end)
            covered += end - start
            last_end = end
        self.assertEqual(covered, total_seqs)


if __name__ == "__main__":
    unittest.main()

