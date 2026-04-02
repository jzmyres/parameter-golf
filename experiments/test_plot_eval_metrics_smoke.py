import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestPlotEvalMetricsSmoke(unittest.TestCase):
    def test_plot_eval_metrics_generates_png(self):
        from experiments.plot_eval_metrics import plot_eval_comparison

        log = "\n".join(
            [
                "step:200/1000 val_loss:3.1 val_bpb:1.50 deq_residual:1.0 deq_iter_conv:0.1 "
                "mlp_ortho:0.25 attn_ortho:0.50 mos_ctp_ortho:0.10 mos_ntp_ortho:0.20 "
                "mlp_cv:0.10 attn_cv:0.10 mos_ctp_cv:0.10 mos_ntp_cv:0.10 "
                "mlp_usage:[0.5,0.5] attn_usage:[0.4,0.6] mos_ctp_usage:[0.33,0.33,0.34] mos_ntp_usage:[0.33,0.33,0.34]",
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

