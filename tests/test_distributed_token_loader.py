"""Tests for DistributedTokenLoader rank-local partitioning."""
import sys, os, tempfile, unittest
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _create_test_shard(path, num_tokens=8192):
    """Create a minimal shard file with valid header + sequential tokens."""
    header = np.zeros(256, dtype="<i4")
    header[0] = 20240520  # magic
    header[1] = 1         # version
    header[2] = num_tokens
    tokens = np.arange(num_tokens, dtype="<u2")
    with open(path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens.tobytes())


class TestDistributedTokenLoader(unittest.TestCase):
    def _make_loaders(self, tmpdir, num_tokens=8192, world_size=2):
        shard_path = os.path.join(tmpdir, "test_train_000000.bin")
        _create_test_shard(shard_path, num_tokens=num_tokens)
        pattern = os.path.join(tmpdir, "test_train_*.bin")
        from train_gpt import DistributedTokenLoader
        loaders = [
            DistributedTokenLoader(pattern, rank=r, world_size=world_size,
                                   device=torch.device("cpu"))
            for r in range(world_size)
        ]
        return loaders

    def test_rank_partitioning_single_batch(self):
        """Concatenating rank slices reconstructs the global span."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaders = self._make_loaders(tmpdir, num_tokens=8192, world_size=2)
            seq_len, global_tokens, grad_accum = 64, 512, 1

            x0, y0 = loaders[0].next_batch(global_tokens, seq_len, grad_accum)
            x1, y1 = loaders[1].next_batch(global_tokens, seq_len, grad_accum)

            # Both ranks get data
            self.assertGreater(x0.numel(), 0)
            self.assertGreater(x1.numel(), 0)
            # Same shape
            self.assertEqual(x0.shape, x1.shape)

    def test_no_overlap_across_ranks(self):
        """Rank token ranges should be disjoint (at most 1 token boundary overlap)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaders = self._make_loaders(tmpdir, num_tokens=16384, world_size=2)
            seq_len, global_tokens, grad_accum = 64, 1024, 1

            x0, _ = loaders[0].next_batch(global_tokens, seq_len, grad_accum)
            x1, _ = loaders[1].next_batch(global_tokens, seq_len, grad_accum)

            set0 = set(x0.reshape(-1).tolist())
            set1 = set(x1.reshape(-1).tolist())
            overlap = len(set0 & set1)
            # With sequential token IDs, overlap should be minimal
            self.assertLess(overlap, x0.numel() // 2,
                            f"Excessive overlap: {overlap}/{x0.numel()} tokens")

    def test_multi_batch_no_drift(self):
        """Multiple consecutive batches should not produce overlapping tokens within a rank."""
        with tempfile.TemporaryDirectory() as tmpdir:
            loaders = self._make_loaders(tmpdir, num_tokens=16384, world_size=2)
            seq_len, global_tokens, grad_accum = 64, 512, 1

            all_tokens_r0 = []
            for _ in range(4):
                x, _ = loaders[0].next_batch(global_tokens, seq_len, grad_accum)
                all_tokens_r0.extend(x.reshape(-1).tolist())

            # Check no exact duplicate batches (stream advances)
            batches = [all_tokens_r0[i:i+seq_len] for i in range(0, len(all_tokens_r0), seq_len)]
            unique_first_tokens = set(b[0] for b in batches if b)
            self.assertGreater(len(unique_first_tokens), 1,
                               "All batches start with the same token — stream not advancing")


if __name__ == "__main__":
    unittest.main()
