import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPlotEvalMetricsSmoke(unittest.TestCase):
    def test_plot_eval_metrics_generates_png(self):
        try:
            import matplotlib  # noqa: F401
        except Exception:
            self.skipTest("matplotlib not available in this environment")
        from experiments.plot_eval_metrics import plot_eval_comparison

        log = "\n".join(
            [
                "step:200/1000 val_loss:3.1 val_bpb:1.50 deq_residual:1.0 deq_iter_conv:0.1 "
                "block_ortho:0.15 mos_ctp_ortho:0.10 mos_ntp_ortho:0.20 "
                "block_cv:0.10 mos_ctp_cv:0.10 mos_ntp_cv:0.10 "
                "block_entropy:0.90 mos_ctp_entropy:1.00 mos_ntp_entropy:1.10 "
                "block_usage:[0.2,0.2,0.2,0.2,0.1,0.1] mos_ctp_usage:[0.33,0.33,0.34] mos_ntp_usage:[0.33,0.33,0.34]",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            b = os.path.join(td, "baseline.log")
            c = os.path.join(td, "current.log")
            outdir = os.path.join(td, "out")
            os.makedirs(outdir, exist_ok=True)
            with open(b, "w", encoding="utf-8") as f:
                f.write(log)
            with open(c, "w", encoding="utf-8") as f:
                f.write(log)
            ok = plot_eval_comparison(b, c, outdir)
            self.assertTrue(ok)
            self.assertTrue(os.path.exists(os.path.join(outdir, "metrics_eval_comparison.png")))


if __name__ == "__main__":
    unittest.main()
