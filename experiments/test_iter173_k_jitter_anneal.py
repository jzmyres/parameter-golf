"""iter173 K-jitter weight annealing curriculum tests (2026-05-17).

Linear interpolation between `deq_k_jitter_weights` (start) and
`deq_k_jitter_weights_final` (end) over training window
[anneal_start_frac, anneal_end_frac] · total_steps.

Hypothesis: iter172 achieved rho_F=0.77 at K=128 — model is "ready" for
deeper K. Shifting K-jitter probability mass toward larger K teaches the
model to use the depth that K-sweep eval actually uses.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_gpt import (
    Hyperparameters,
    KShuffleBagSampler,
    _compute_annealed_k_jitter_weights,
)
import random


class TestAnnealHelper(unittest.TestCase):
    INITIAL = (0.50, 0.40, 0.07, 0.03, 0.015, 0.0075)
    FINAL = (0.125, 0.125, 0.125, 0.125, 0.25, 0.25)

    def test_disabled_when_final_empty(self):
        """Default Hyperparameters.deq_k_jitter_weights_final=() → no anneal."""
        w = _compute_annealed_k_jitter_weights(
            initial=self.INITIAL, final=(), step=500, total_steps=1000,
            start_frac=0.3, end_frac=0.9,
        )
        self.assertEqual(w, self.INITIAL)

    def test_disabled_when_length_mismatch(self):
        """Annealing requires same-length initial/final tuples (defensive)."""
        w = _compute_annealed_k_jitter_weights(
            initial=self.INITIAL, final=(0.5, 0.5), step=500, total_steps=1000,
            start_frac=0.3, end_frac=0.9,
        )
        self.assertEqual(w, self.INITIAL)

    def test_pre_anneal_window_returns_initial(self):
        """Before start_frac · total_steps, no shift toward final."""
        w = _compute_annealed_k_jitter_weights(
            initial=self.INITIAL, final=self.FINAL, step=200, total_steps=1000,
            start_frac=0.3, end_frac=0.9,
        )
        self.assertEqual(w, self.INITIAL)

    def test_post_anneal_window_returns_final(self):
        """After end_frac · total_steps, fully at final distribution."""
        w = _compute_annealed_k_jitter_weights(
            initial=self.INITIAL, final=self.FINAL, step=950, total_steps=1000,
            start_frac=0.3, end_frac=0.9,
        )
        self.assertEqual(w, self.FINAL)

    def test_midpoint_interpolation(self):
        """At anneal midpoint (60% through training), weights are 50/50."""
        # start=300, end=900, midpoint=600
        w = _compute_annealed_k_jitter_weights(
            initial=self.INITIAL, final=self.FINAL, step=600, total_steps=1000,
            start_frac=0.3, end_frac=0.9,
        )
        for w_t, i, f in zip(w, self.INITIAL, self.FINAL):
            self.assertAlmostEqual(w_t, 0.5 * i + 0.5 * f, places=5)

    def test_anneal_shifts_expected_K_toward_deeper(self):
        """The promotion-relevant property: E[K] at end > E[K] at start."""
        K_set = (16, 24, 32, 64, 96, 128)
        e_k_init = sum(k * w for k, w in zip(K_set, self.INITIAL))
        e_k_final = sum(k * w for k, w in zip(K_set, self.FINAL))
        self.assertLess(e_k_init, e_k_final,
            f"iter173 must shift K-jitter probability mass toward larger K: "
            f"E[K]_init={e_k_init:.1f} must be < E[K]_final={e_k_final:.1f}")


class TestSamplerSetWeights(unittest.TestCase):
    def test_set_weights_clears_bag_and_resamples(self):
        """KShuffleBagSampler.set_weights() must clear cached bag so the next
        sample() draws from the new distribution."""
        values = [16, 32, 128]
        initial_weights = [0.95, 0.04, 0.01]  # heavily K=16
        sampler = KShuffleBagSampler(
            16, 128, random.Random(42), values=values, weights=initial_weights,
        )
        # Draw many samples to confirm initial distribution
        samples_initial = [sampler.sample() for _ in range(1000)]
        n_k16_initial = sum(1 for k in samples_initial if k == 16)
        self.assertGreater(n_k16_initial, 800,
            f"Initial weights heavily favor K=16; got only {n_k16_initial}/1000")

        # Set new weights: heavily K=128
        new_weights = [0.01, 0.04, 0.95]
        sampler.set_weights(new_weights)
        samples_after = [sampler.sample() for _ in range(1000)]
        n_k128_after = sum(1 for k in samples_after if k == 128)
        self.assertGreater(n_k128_after, 800,
            f"After set_weights, K=128 must dominate; got only {n_k128_after}/1000")

    def test_set_weights_rejects_no_values(self):
        """set_weights without explicit values is undefined."""
        sampler = KShuffleBagSampler(16, 128, random.Random(42))
        with self.assertRaises(ValueError):
            sampler.set_weights([0.5, 0.5])


class TestHyperparameterDefaults(unittest.TestCase):
    def test_anneal_defaults_disabled(self):
        """iter173 default OFF — empty final tuple preserves iter172 behavior."""
        self.assertEqual(Hyperparameters.deq_k_jitter_weights_final, ())
        self.assertEqual(Hyperparameters.deq_k_jitter_anneal_start_frac, 0.3)
        self.assertEqual(Hyperparameters.deq_k_jitter_anneal_end_frac, 0.9)


if __name__ == "__main__":
    unittest.main()
