import os
import sys
import tempfile
import unittest
import math


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from experiments.plot_metrics import parse_log, usage_min_series


class TestPlotMetricsParse(unittest.TestCase):
    def test_parse_log_train_diagnostics_fields(self):
        log = "\n".join(
            [
                # Train log line includes diagnostics we want plotted densely.
                "step:10/20 train_loss:3.2 ntp_loss:2.1 ctp_loss:1.1 grad_norm:0.9 "
                "train_time:10.0ms step_avg:10.0ms "
                "deq_residual:1.0 deq_recon_err:0.0 deq_iter_conv:0.1 gg_iter:[0.9,0.8,0.7,0.6] "
                "attn_rg_iter:[0.99,0.98,0.97,0.96] mlp_rg_iter:[0.88,0.87,0.86,0.85] "
                "mlp_ortho:0.25 attn_ortho:0.50 mos_ctp_ortho:0.10 mos_ntp_ortho:0.20 "
                "mlp_entropy:0.7 attn_entropy:0.6 mos_ctp_entropy:0.8 mos_ntp_entropy:0.9 "
                "mlp_cv:0.01 attn_cv:0.02 mos_ctp_cv:0.03 mos_ntp_cv:0.04 "
                "mlp_usage:[0.5,0.5] attn_usage:[0.4,0.6] mos_ctp_usage:[0.1,0.9,0.0] mos_ntp_usage:[0.2,0.8,0.0]",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(d["train_steps"], [10])
        self.assertEqual(d["deq_residual_train"], [1.0])
        self.assertEqual(d["deq_recon_train"], [0.0])
        self.assertEqual(d["deq_iter_conv_train"], [0.1])
        self.assertEqual(d["gg_iter_train"], [[0.9, 0.8, 0.7, 0.6]])
        self.assertEqual(d["attn_rg_iter_train"], [[0.99, 0.98, 0.97, 0.96]])
        self.assertEqual(d["mlp_rg_iter_train"], [[0.88, 0.87, 0.86, 0.85]])
        self.assertEqual(d["mlp_ortho_train"], [0.25])
        self.assertEqual(d["attn_ortho_train"], [0.50])
        self.assertEqual(d["mos_ctp_ortho_train"], [0.10])
        self.assertEqual(d["mos_ntp_ortho_train"], [0.20])
        self.assertEqual(usage_min_series(d, "mlp_usage_train"), [0.5])
        self.assertEqual(usage_min_series(d, "mos_ctp_usage_train"), [0.0])

    def test_parse_log_missing_numeric_metrics_become_nan(self):
        log = "\n".join(
            [
                # val line missing deq/expert metrics should not be silently treated as 0.
                "step:1/10 val_loss:3.1 val_bpb:1.50 train_time:10ms step_avg:10.0ms",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(d["val_steps"], [1])
        self.assertTrue(math.isnan(d["deq_residual"][-1]))
        self.assertTrue(math.isnan(d["mlp_entropy"][-1]))

    def test_parse_log_orthogonality_fields(self):
        log = "\n".join(
            [
                "step:1/10 train_loss:3.2 ntp_loss:2.1 ctp_loss:1.1 grad_norm:0.9 train_time:10.0ms step_avg:10.0ms",
                "step:1/10 val_loss:3.1 val_bpb:1.50 train_time:10ms step_avg:10.0ms "
                "deq_residual:1.0 deq_recon_err:0.0 deq_iter_conv:0.1 gg_iter:[0.1,0.2] "
                "mlp_ortho:0.25 attn_ortho:0.50 mos_ctp_ortho:0.10 mos_ntp_ortho:0.20 expert_ortho:0.25 "
                "mlp_entropy:0.7 attn_entropy:0.6 mos_ctp_entropy:0.8 mos_ntp_entropy:0.9 "
                "mlp_usage:[0.5,0.5] attn_usage:[0.4,0.6] mos_ctp_usage:[0.1,0.9,0.0] mos_ntp_usage:[0.2,0.8,0.0]",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(d["val_steps"], [1])
        self.assertEqual(d["mlp_ortho"][-1], 0.25)
        self.assertEqual(d["attn_ortho"][-1], 0.50)
        self.assertEqual(d["mos_ctp_ortho"][-1], 0.10)
        self.assertEqual(d["mos_ntp_ortho"][-1], 0.20)
        self.assertEqual(d["expert_ortho"][-1], 0.25)
        self.assertEqual(d["gg_iter"][-1], [0.1, 0.2])

        self.assertEqual(usage_min_series(d, "mlp_usage"), [0.5])
        self.assertEqual(usage_min_series(d, "mos_ctp_usage"), [0.0])

    def test_parse_log_legacy_expert_ortho_only(self):
        log = "\n".join(
            [
                "step:1/10 val_loss:3.1 val_bpb:1.50 train_time:10ms step_avg:10.0ms expert_ortho:0.33",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(d["expert_ortho"], [0.33])


if __name__ == "__main__":
    unittest.main()
