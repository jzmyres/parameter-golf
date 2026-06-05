import os
import sys
import tempfile
import unittest
import math


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from experiments.plot_metrics import parse_log, plot_comparison, usage_min_series


class TestPlotMetricsParse(unittest.TestCase):
    def test_parse_log_concatenated_runs_keep_last(self):
        log = "\n".join(
            [
                # Run 1
                "train_batch_tokens:8 train_seq_len:4 iterations:2 warmup_steps:0 max_wallclock_seconds:0.000",
                "step:1/2 train_loss:3.2 ntp_loss:2.1 ctp_loss:1.1 grad_norm:0.9 train_time:10.0ms step_avg:10.0ms",
                "step:1/2 val_loss:3.1 val_bpb:1.50 deq_residual:1.0 mlp_entropy:0.7",
                "final_int6_zstd_roundtrip_exact val_loss:2.00000000 val_bpb:1.40000000",
                # Run 2 (concatenated into same file)
                "train_batch_tokens:8 train_seq_len:4 iterations:2 warmup_steps:0 max_wallclock_seconds:0.000",
                "step:1/2 train_loss:2.2 ntp_loss:1.1 ctp_loss:0.1 grad_norm:0.1 train_time:5.0ms step_avg:5.0ms",
                "step:1/2 val_loss:2.1 val_bpb:1.25 deq_residual:2.0 mlp_entropy:0.8",
                "final_int6_zstd_roundtrip_exact val_loss:1.90000000 val_bpb:1.35000000",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        # Only the second run should remain.
        self.assertEqual(d["train_steps"], [1])
        self.assertEqual(d["train_loss"], [2.2])
        self.assertEqual(len(d["router_reg_loss"]), 1)
        self.assertTrue(math.isnan(d["router_reg_loss"][0]))
        # router_cv_term / mos_cv_term removed from derived metrics 2026-05-15.
        self.assertEqual(len(d["parcae_a_bar_mean"]), 1)
        self.assertTrue(math.isnan(d["parcae_a_bar_mean"][0]))
        self.assertEqual(d["val_steps"], [1])
        self.assertEqual(d["val_bpb"], [1.25])
        self.assertEqual(d["final_postquant_val_bpb"], 1.35)

    def test_parse_log_train_diagnostics_fields(self):
        log = "\n".join(
            [
                # Train log line includes diagnostics we want plotted densely.
                "step:10/20 train_loss:3.2 ntp_loss:2.1 ctp_loss:1.1 grad_norm:0.9 "
                "router_cv_loss:0.120000 router_pertoken_entropy_loss:2.300000 mos_cv_loss:0.040000 "
                "expert_diversity_loss:0.500000 mos_diversity_loss:0.000000 router_reg_loss:0.091625 "
                "router_pertoken_entropy_coef_eff:0.00125 "
                "expert_diversity_coef_eff:0.0375 mos_diversity_coef_eff:0 "
                "parcae_a_bar_min:0.600000 parcae_a_bar_mean:0.700000 parcae_a_bar_max:0.800000 "
                "parcae_a_bar_core_max:0.777778 parcae_beta_mean:0.300000 parcae_beta_max:0.400000 "
                "parcae_b_bar_mean:0.300000 parcae_b_bar_max:0.310000 "
                "parcae_delta_mean:1.000000 parcae_delta_max:1.010000 "
                "parcae_recon_amp_log10:3.550000 "
                "train_time:10.0ms step_avg:10.0ms "
                "deq_residual:1.0 deq_recon_err:0.0 tbptt_recon:0.0 deq_fp_travel:0.42 deq_iter_conv:0.1 gg_iter:[0.9,0.8,0.7,0.6] "
                "expert_ortho:0.25 mos_ctp_ortho:0.10 mos_ntp_ortho:0.20 "
                "block_entropy:0.65 block_cv:0.015 block_usage:[0.45,0.55] "
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
        self.assertEqual(d["tbptt_recon_train"], [0.0])
        self.assertAlmostEqual(d["deq_fp_travel_train"][0], 0.42, places=5)
        self.assertEqual(d["deq_iter_conv_train"], [0.1])
        self.assertEqual(d["gg_iter_train"], [[0.9, 0.8, 0.7, 0.6]])
        self.assertEqual(d["router_cv_loss"], [0.12])
        self.assertEqual(d["router_pertoken_entropy_loss"], [2.3])
        self.assertEqual(d["mos_cv_loss"], [0.04])
        self.assertEqual(d["expert_diversity_loss"], [0.5])
        self.assertEqual(d["mos_diversity_loss"], [0.0])
        self.assertEqual(d["router_reg_loss"], [0.091625])
        self.assertEqual(d["router_pertoken_entropy_coef_eff"], [0.00125])
        self.assertEqual(d["expert_diversity_coef_eff"], [0.0375])
        self.assertEqual(d["mos_diversity_coef_eff"], [0.0])
        # router_cv_term / mos_cv_term removed 2026-05-15 (CV-as-loss
        # removed; CV remains as diagnostic only).
        self.assertAlmostEqual(d["router_pertoken_entropy_term"][0], 0.002875)
        self.assertAlmostEqual(d["expert_diversity_term"][0], 0.01875)
        self.assertAlmostEqual(d["mos_diversity_term"][0], 0.0)
        self.assertEqual(d["parcae_a_bar_min"], [0.6])
        self.assertEqual(d["parcae_a_bar_mean"], [0.7])
        self.assertEqual(d["parcae_a_bar_max"], [0.8])
        self.assertAlmostEqual(d["parcae_a_bar_core_max"][0], 0.777778, places=5)
        self.assertEqual(d["parcae_beta_mean"], [0.3])
        self.assertEqual(d["parcae_beta_max"], [0.4])
        self.assertEqual(d["parcae_b_bar_mean"], [0.3])
        self.assertEqual(d["parcae_b_bar_max"], [0.31])
        self.assertEqual(d["parcae_delta_mean"], [1.0])
        self.assertEqual(d["parcae_delta_max"], [1.01])
        self.assertEqual(d["parcae_recon_amp_log10"], [3.55])
        self.assertEqual(d["mos_ctp_ortho_train"], [0.10])
        self.assertEqual(d["mos_ntp_ortho_train"], [0.20])
        self.assertEqual(d["block_entropy_train"], [0.65])
        self.assertEqual(d["block_cv_train"], [0.015])
        self.assertEqual(usage_min_series(d, "block_usage_train"), [0.45])
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

    def test_parse_k_sweep_effective_depth_fields(self):
        log = "\n".join(
            [
                "k_sweep:k=16 val_bpb:1.234500 ED_update:3.2500 ED_logit:2.5000 "
                "route_depth_nmi_mean:0.1200 route_depth_nmi_max:0.2000 "
                "expert_util_mean:0.9000 expert_output_erank_mean:7.0000 "
                "rho_F:0.8000 sigma_max_F:0.9700 fp_residual_F:0.001000",
                "k_sweep_table:    K   val_bpb ED_update  ED_logit route_depth_nmi_mean route_depth_nmi_max expert_util_mean expert_output_erank_mean rho_F sigma_max_F fp_residual_F iter_conv_rel",
                "k_sweep_table:   16    1.2345    3.2500    2.5000               0.1200              0.2000           0.9000                   7.0000 0.8000      0.9700       0.001000       0.0100",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(d["k_sweep"][0]["ED_update"], 3.25)
        self.assertEqual(d["k_sweep"][0]["route_depth_nmi_mean"], 0.12)
        self.assertEqual(d["k_sweep"][0]["sigma_max_F"], 0.97)
        self.assertEqual(d["k_sweep_table"][0]["ED_logit"], 2.5)
        self.assertEqual(d["k_sweep_table"][0]["expert_output_erank_mean"], 7.0)

    def test_parse_log_orthogonality_fields(self):
        log = "\n".join(
            [
                "step:1/10 train_loss:3.2 ntp_loss:2.1 ctp_loss:1.1 grad_norm:0.9 train_time:10.0ms step_avg:10.0ms",
                "step:1/10 val_loss:3.1 val_bpb:1.50 train_time:10ms step_avg:10.0ms "
                "deq_residual:1.0 deq_recon_err:0.0 tbptt_recon:0.0 deq_iter_conv:0.1 gg_iter:[0.1,0.2] "
                "block_ortho:0.27 expert_ortho:0.25 mos_ctp_ortho:0.10 mos_ntp_ortho:0.20 "
                "block_entropy:0.65 block_cv:0.015 block_usage:[0.45,0.55] "
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
        self.assertEqual(d["mos_ctp_ortho"][-1], 0.10)
        self.assertEqual(d["mos_ntp_ortho"][-1], 0.20)
        self.assertEqual(d["block_ortho"][-1], 0.27)
        self.assertEqual(d["expert_ortho"][-1], 0.25)
        self.assertEqual(d["gg_iter"][-1], [0.1, 0.2])
        self.assertEqual(d["block_entropy"][-1], 0.65)
        self.assertEqual(d["block_cv"][-1], 0.015)
        self.assertEqual(usage_min_series(d, "block_usage"), [0.45])

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

    def test_parse_m0_two_goal_metrics_line(self):
        from experiments.plot_metrics import M0_METRICS_FIELDS

        log = "\n".join(
            [
                "step:10/20 k_hi:64 train_loss:3.2000",
                # peak_vram is the PRIMARY resource field; active_frac is now the
                # MoE-mechanism diagnostic (emitted with a diag: prefix).
                "metrics: erank:512.3456 peak_vram:1024.0000 kv_bytes:128 params:1234567 "
                "router_entropy:2.0794 expert_util:1.9000 route_step_div:0.6250 "
                "expert_cos_div:0.7000 disp_tail:0.3300 "
                "ops:tok_per_s:50000.0000 "
                "ops:vram_util_pct:1.2500 diag:active_frac:0.7500",
                # Control-sweep line carrying the sparse R_act / phi fields too.
                "step:20/20 k_hi:64 train_loss:3.0000",
                "metrics: erank:600.0000 peak_vram:2048.0000 kv_bytes:128 params:1234567 "
                "router_entropy:1.5000 expert_util:2.5000 route_step_div:0.8000 "
                "expert_cos_div:0.9000 disp_tail:0.4100 "
                "ops:tok_per_s:60000.0000 "
                "ops:vram_util_pct:2.5000 R_act:2.5000 phi:0.8000 diag:active_frac:1.0000",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        # Every registered M0 metric field is present in the parsed dict.
        for field in M0_METRICS_FIELDS:
            self.assertIn(field, d)
            self.assertEqual(len(d[field]), 2)

        self.assertEqual(d["metrics_steps"], [10, 20])
        self.assertEqual(d["erank"], [512.3456, 600.0])
        self.assertEqual(d["peak_vram"], [1024.0, 2048.0])
        self.assertEqual(d["active_frac"], [0.75, 1.0])
        self.assertEqual(d["router_entropy"], [2.0794, 1.5])
        self.assertEqual(d["expert_util"], [1.9, 2.5])
        # MoE-basis-depth diagnostics (across-step route diversity + expert
        # output-cosine diversity).
        self.assertEqual(d["route_step_div"], [0.625, 0.8])
        self.assertEqual(d["expert_cos_div"], [0.7, 0.9])
        self.assertEqual(d["disp_tail"], [0.33, 0.41])
        self.assertEqual(d["kv_bytes"], [128.0, 128.0])
        self.assertEqual(d["params"], [1234567.0, 1234567.0])
        # Ops/efficiency diagnostics (NOT resource-goal numbers): throughput
        # (tokens/wall-second) and VRAM utilization (% of device memory).
        self.assertEqual(d["tok_per_s"], [50000.0, 60000.0])
        self.assertEqual(d["vram_util_pct"], [1.25, 2.5])
        # R_act / phi absent on the first line -> NaN; present on the second.
        self.assertTrue(math.isnan(d["R_act"][0]))
        self.assertTrue(math.isnan(d["phi"][0]))
        self.assertEqual(d["R_act"][1], 2.5)
        self.assertEqual(d["phi"][1], 0.8)

    def test_parse_m0_metrics_recon_rel(self):
        """``recon_rel`` (reversible round-trip / BPTT gradient-correctness gate)
        is a per-step series on the ``metrics:`` line, parsed from sci-notation
        and NaN-tolerant (absent on a line -> NaN, like the other sparse fields)."""
        from experiments.plot_metrics import M0_METRICS_FIELDS

        self.assertIn("recon_rel", M0_METRICS_FIELDS)
        log = "\n".join(
            [
                "step:10/20 k_hi:64 train_loss:3.2000",
                # Scientific-notation value (recon_rel spans 1e-15..1e-2).
                "metrics: erank:512.3456 peak_vram:1024.0000 kv_bytes:128 params:1234567 "
                "disp_tail:0.3300 recon_rel:5.00e-16",
                # Second line OMITS recon_rel -> parses to NaN.
                "step:20/20 k_hi:64 train_loss:3.0000",
                "metrics: erank:600.0000 peak_vram:2048.0000 kv_bytes:128 params:1234567 "
                "disp_tail:0.4100",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(len(d["recon_rel"]), 2)
        self.assertAlmostEqual(d["recon_rel"][0], 5.00e-16)
        self.assertTrue(math.isnan(d["recon_rel"][1]))

    def test_parse_depth_gain_sweep_lines(self):
        """The end-of-run depth-gain MEASUREMENT lines (--k-eval-sweep) parse as
        standalone scalars: depth_sweep -> {K: (val_bpb, val_loss)} map,
        depth_gain_GT / phi_eval -> run-level scalars. NaN-tolerant (synthetic
        smoke emits `nan` for bpb)."""
        log = "\n".join(
            [
                "train_batch_tokens:8 train_seq_len:16 iterations:2 warmup_steps:0 max_wallclock_seconds:0.000",
                "final val_loss:2.5000 val_bpb:1.4000",
                "depth_sweep: K=8 val_bpb:1.4500 val_loss:2.5500",
                "depth_sweep: K=16 val_bpb:1.4200 val_loss:2.5200",
                "depth_sweep: K=64 val_bpb:1.4000 val_loss:2.5000",
                "depth_gain_GT:0.0500",
                # phi_eval carries a trailing "# EVAL-K proxy ..." comment in the
                # real emit; the parser must read the float and ignore the suffix.
                "phi_eval:0.3300  # EVAL-K proxy (NOT phi_isodepth)",
                # PRINCIPLED EXPRESSIVENESS: per-K usable_info (I_V in bits) plus
                # run-level expressiveness_rho / iv_total_bits. Here loss DECREASES
                # with K -> I_V INCREASES -> rho = +1, iv_total_bits > 0.
                "usable_info: K=8 iv_bits:2.0000 val_loss:2.5500",
                "usable_info: K=16 iv_bits:2.0500 val_loss:2.5200",
                "usable_info: K=64 iv_bits:2.1000 val_loss:2.5000",
                "expressiveness_rho:1.0000",
                "iv_total_bits:0.1000",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(set(d["depth_sweep"].keys()), {8, 16, 64})
        self.assertEqual(d["depth_sweep"][8], (1.45, 2.55))
        self.assertEqual(d["depth_sweep"][64], (1.40, 2.50))
        self.assertEqual(d["depth_gain_GT"], 0.05)
        self.assertEqual(d["phi_eval"], 0.33)
        # usable_info -> {K: iv_bits} map; rho / iv_total_bits -> run scalars.
        self.assertEqual(set(d["usable_info"].keys()), {8, 16, 64})
        self.assertEqual(d["usable_info"][8], 2.0)
        self.assertEqual(d["usable_info"][64], 2.1)
        self.assertEqual(d["expressiveness_rho"], 1.0)
        self.assertEqual(d["iv_total_bits"], 0.1)

    def test_parse_phi_isodepth_train_r_sweep_lines(self):
        """The PRINCIPLED Iso-Depth harness (experiments/measure_phi.py) emits a
        per-r `phi_sweep:` line and a final `phi_isodepth:` scalar. phi_sweep ->
        {r: (val_loss, val_bpb)} map; phi_isodepth -> the fitted exponent."""
        log = "\n".join(
            [
                "measure_phi: r_list=[1, 2, 4, 8, 16] n_once=2000000 n_rec=6000000",
                "phi_sweep: r=1 val_loss:2.9200 val_bpb:1.6000",
                "phi_sweep: r=2 val_loss:2.9000 val_bpb:1.5800",
                "phi_sweep: r=4 val_loss:2.8800 val_bpb:1.5600",
                "phi_sweep: r=8 val_loss:2.8600 val_bpb:1.5400",
                "phi_sweep: r=16 val_loss:2.8400 val_bpb:1.5200",
                ("phi_isodepth: 0.4600 alpha:0.3000 E:2.5000 rmse:1.2e-08 "
                 "n_once:2000000 n_rec:6000000 n_points:5"),
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(set(d["phi_sweep"].keys()), {1, 2, 4, 8, 16})
        self.assertEqual(d["phi_sweep"][1], (2.92, 1.60))
        self.assertEqual(d["phi_sweep"][16], (2.84, 1.52))
        self.assertEqual(d["phi_isodepth"], 0.46)

    def test_parse_phi_isodepth_nan_and_failed_run_tolerant(self):
        """A degenerate fit emits `phi_isodepth: nan ...` and a crashed r prints
        `val_loss:FAILED` (skipped by the float-only phi_sweep regex)."""
        log = "\n".join(
            [
                "phi_sweep: r=1 val_loss:2.9200 val_bpb:nan",
                "phi_sweep: r=2 val_loss:FAILED val_bpb:FAILED",
                "phi_isodepth: nan alpha:nan E:nan rmse:nan n_once:10 n_rec:20 "
                "n_points:1 reason:need>=3 finite points, got 1",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        # Only the finite-float r=1 line is captured; the FAILED r=2 is skipped.
        self.assertEqual(set(d["phi_sweep"].keys()), {1})
        self.assertTrue(math.isnan(d["phi_sweep"][1][1]))  # val_bpb nan
        self.assertTrue(math.isnan(d["phi_isodepth"]))

    def test_parse_depth_gain_sweep_nan_tolerant(self):
        """On the synthetic smoke (no tokenizer) bpb is `nan`; the parser must
        not choke and must store NaN for bpb / depth_gain_GT while keeping the
        finite val_loss column."""
        log = "\n".join(
            [
                "train_batch_tokens:8 train_seq_len:16 iterations:2 warmup_steps:0 max_wallclock_seconds:0.000",
                "depth_sweep: K=2 val_bpb:nan val_loss:3.4500",
                "depth_sweep: K=4 val_bpb:nan val_loss:3.4000",
                "depth_gain_GT:nan",
                "phi_eval:0.0000",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(set(d["depth_sweep"].keys()), {2, 4})
        self.assertTrue(math.isnan(d["depth_sweep"][2][0]))
        self.assertEqual(d["depth_sweep"][2][1], 3.45)
        self.assertTrue(math.isnan(d["depth_gain_GT"]))
        self.assertEqual(d["phi_eval"], 0.0)

    def test_parse_usable_info_nan_and_negative_rho_tolerant(self):
        """expressiveness_rho is NaN with <3 finite K; an anti-expressive run
        (loss increasing with K -> I_V decreasing) emits a negative rho and a
        negative iv_total_bits. The parser must capture both faithfully."""
        log = "\n".join(
            [
                "train_batch_tokens:8 train_seq_len:16 iterations:2 warmup_steps:0 max_wallclock_seconds:0.000",
                # Anti-expressive: I_V DECREASES as K grows.
                "usable_info: K=2 iv_bits:2.1000 val_loss:3.4000",
                "usable_info: K=4 iv_bits:2.0000 val_loss:3.5000",
                "expressiveness_rho:nan",
                "iv_total_bits:-0.1000",
            ]
        )
        with tempfile.TemporaryDirectory() as td:
            p = os.path.join(td, "log.txt")
            with open(p, "w", encoding="utf-8") as f:
                f.write(log)
            d = parse_log(p)

        self.assertEqual(set(d["usable_info"].keys()), {2, 4})
        self.assertEqual(d["usable_info"][2], 2.1)
        self.assertTrue(math.isnan(d["expressiveness_rho"]))
        self.assertEqual(d["iv_total_bits"], -0.1)

    def test_plot_comparison_smoke_with_auxiliary_terms(self):
        train_line = (
            "step:10/20 train_loss:3.2 ntp_loss:2.1 ctp_loss:1.1 grad_norm:0.9 "
            "router_cv_loss:0.120000 router_pertoken_entropy_loss:2.300000 mos_cv_loss:0.040000 "
            "expert_diversity_loss:0.500000 mos_diversity_loss:0.000000 router_reg_loss:0.091625 "
            "router_pertoken_entropy_coef_eff:0.00125 "
            "expert_diversity_coef_eff:0.0375 mos_diversity_coef_eff:0 "
            "parcae_a_bar_min:0.600000 parcae_a_bar_mean:0.700000 parcae_a_bar_max:0.800000 "
            "parcae_a_bar_core_max:0.777778 parcae_beta_mean:0.300000 parcae_beta_max:0.400000 "
            "parcae_b_bar_mean:0.300000 parcae_b_bar_max:0.310000 "
            "parcae_delta_mean:1.000000 parcae_delta_max:1.010000 "
            "parcae_recon_amp_log10:3.550000 "
            "train_time:10.0ms step_avg:10.0ms"
        )
        val_line = "step:10/20 val_loss:3.1 val_bpb:1.50 train_time:10ms step_avg:10.0ms"
        log = "\n".join([train_line, val_line, "final_int6_zstd_roundtrip_exact val_loss:3.0 val_bpb:1.49"])
        with tempfile.TemporaryDirectory() as td:
            baseline = os.path.join(td, "baseline.log")
            current = os.path.join(td, "current.log")
            with open(baseline, "w", encoding="utf-8") as f:
                f.write(log)
            with open(current, "w", encoding="utf-8") as f:
                f.write(log)
            ok = plot_comparison(baseline, current, td)
            if not ok:
                self.skipTest("matplotlib not available")
            self.assertTrue(os.path.exists(os.path.join(td, "metrics_comparison.png")))


if __name__ == "__main__":
    unittest.main()
