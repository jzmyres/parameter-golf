import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_GPT = ROOT / "legacy" / "train_gpt_rich.py"
UPDATE_RESULTS = ROOT / "experiments" / "update_results.sh"
OPG_DOC = ROOT / "reports" / "opg_doc.tex"
P1_SYNTHETIC = ROOT / "experiments" / "p1_synthetic.py"
P1_SYNTHETIC_PIPELINE = ROOT / "experiments" / "run_p1_synthetic_pipeline.sh"
P1_FEEDBACK_STAGES = ROOT / "experiments" / "p1_feedback_stages.py"
P1_FEEDBACK_PIPELINE = ROOT / "experiments" / "run_feedback_stage_pipeline.sh"
CLAUDE = ROOT / "CLAUDE.md"
EXPERIENCE = ROOT / "EXPERIENCE.md"
EXPERIMENT_DOCS = ROOT / "experiments" / "docs"
LEGACY_DOCS = ROOT / "legacy" / "docs"


def _hot_path_body(text: str) -> str:
    body = text.split("for micro_step in range(grad_accum_steps):", 1)[1]
    return body.split("train_loss /= grad_accum_steps", 1)[0]


class TestTrainingContracts(unittest.TestCase):
    def test_default_training_budget_is_step_count_governed(self) -> None:
        """Default runs: 1000 iterations on DDP with all GPUs; wallclock off.

        Submission runs OVERRIDE via `--max-training-seconds=600` (8xH100
        training budget). The source-of-truth default is step-count,
        not wallclock — so Hyperparameters defaults must be:
            iterations = 1000
            max_wallclock_seconds = 0
        """
        text = TRAIN_GPT.read_text()
        # Match the Hyperparameters class attribute assignments anywhere in the file.
        self.assertRegex(text, r"\n\s*iterations\s*=\s*1000\b",
                         "Hyperparameters.iterations default must be 1000")
        self.assertRegex(text, r"\n\s*max_wallclock_seconds\s*=\s*0\b",
                         "Hyperparameters.max_wallclock_seconds default must be 0 "
                         "(disabled; step-count governs). Submission runs set 600 via CLI.")

    def test_gradient_hot_path_has_no_cpu_syncs(self) -> None:
        body = _hot_path_body(TRAIN_GPT.read_text())
        violations = re.findall(r"\.(?:item|cpu|tolist|numpy)\s*\(", body)
        self.assertEqual(violations, [])

    def test_gradient_hot_path_has_no_tensor_to_python_casts(self) -> None:
        body = _hot_path_body(TRAIN_GPT.read_text())
        casts = re.findall(r"\b(?:float|int|bool)\s*\([^)\n]*\)", body)
        offenders = [c for c in casts if re.search(r"_t\b|\.detach\b|tensor|Tensor|\.sum\b|\.mean\b|\.norm\b", c)]
        self.assertEqual(offenders, [],
                         "float()/int()/bool() on a tensor in the hot path — use torch ops.")

    def test_lyapunov_penalty_targets_finite_expansion(self) -> None:
        body = _hot_path_body(TRAIN_GPT.read_text())
        self.assertIn("lyap_loss = torch.relu(expansion - float(base_model.lyapunov_gamma)).pow(2)", body,
                      "Lyapunov penalty must directly constrain finite expansion of T_theta.")
        self.assertNotRegex(body, r"\bif\s+scale_[^\n:]*:",
                            "No Python branch on tensor scale values allowed in the hot path.")

    def test_final_metadata_requires_full_validation_for_promotion(self) -> None:
        text = TRAIN_GPT.read_text()
        self.assertRegex(text, r"\n\s*final_full_validation\s*=\s*True\b")
        # After P0-3 the inline `meta_json["status"] = ...` literal moved into
        # `_compute_run_status`; the contract is now pinned by the helper's
        # return values plus the focused truth-table test in
        # `tests/test_hyperparameter_validation.py`.
        self.assertIn('"validated_fast_only"', text)
        self.assertIn('"final_full_validation_disabled"', text)
        self.assertIn('_compute_run_status(', text)

    def test_refuted_pure_ift_path_is_removed(self) -> None:
        # Word-boundary regex so the assertion fires on actual symbol
        # re-introduction, not on incidental substrings inside identifiers,
        # comments, or unrelated tokens (e.g. `deq_ifteration`, `gift_iters`).
        text = TRAIN_GPT.read_text()
        for pattern in (r"\bdeq_ift\b", r"\b_backward_ift\b", r"\bift_iters\b"):
            self.assertIsNone(
                re.search(pattern, text),
                f"refuted IFT identifier matching {pattern!r} re-introduced",
            )

    def test_opg_doc_tracks_final_minimal_p1_design(self) -> None:
        text = OPG_DOC.read_text()
        for snippet in (
            "Final Minimal Design",
            "P1, primary",
            "P2, memory",
            "P3, secondary",
            r"Core \(M_0\)",
            r"\(M_{\rm clk}\)",
            "Positive control",
            "additive-coupling reversible recurrence",
            "reversible full BPTT",
            "Tier 1",
            r"\(S_5\)",
            "Barrington",
            "experiments/p1\\_synthetic.py",
            "Pipeline Simplification Audit",
            "Current Simplified S+0 DDP Verification",
            "gpu67\\_s0\\_pruned\\_i1000",
        ):
            self.assertIn(snippet, text)
        for metric in ("G_T", r"\mathrm{NDR}_\epsilon", "Pareto"):
            self.assertIn(metric, text)
        # Mechanisms that stay EXCLUDED from the P1 science path even under the
        # retained-M0 design (FP/diagnostic/complexity cruft).
        for snippet in (
            "Dirichlet--UCB router",
            "int6 quantization",
            "Lyapunov pressure",
            "excluded from the P1 science path",
        ):
            self.assertIn(snippet, text)

    def test_opg_doc_retains_mla_moe_mos_mechanisms(self) -> None:
        r"""opg_doc.tex must describe MLA, MoE (smooth routing), and MoS as
        RETAINED resource/expressiveness mechanisms — not future/excluded.

        The M0 design retains MLA (low-rank KV resource), smooth-sparse MoE
        (effective-depth basis), and MoS (output-rank expressiveness). The
        \S{}Goal exclusion list and \S{}Models readout were realigned to this.
        Dirichlet--UCB, DEQ/implicit backward, fixed-point consistency, and
        spectral/Lyapunov pressure stay EXCLUDED.
        """
        text = OPG_DOC.read_text()
        lowered = text.lower()
        # MLA / MoE / MoS are now retained mechanisms with the resource and
        # expressiveness justification language.
        self.assertIn("retained for the resource", lowered)
        self.assertIn("expressiveness", lowered)
        for retained in (
            "MLA",
            "mixture-of-softmaxes",
            "MoS",
        ):
            self.assertIn(retained, text)
        # Smooth / non-top-k MoE routing (the reversibility-safe form).
        self.assertRegex(lowered, r"smooth[- ]sparse|smooth.{0,40}moe|relu.{0,20}softmax")
        # Excluded complexity/FP/diagnostic mechanisms stay excluded.
        for excluded in (
            "Dirichlet--UCB",
            "fixed-point consistency",
        ):
            self.assertIn(excluded, text)
        self.assertRegex(lowered, r"implicit/deq|deq.{0,20}backward|implicit backward")
        self.assertIn("lyapunov pressure", lowered)
        # Prior-evidence / KV-axis staging framing remains coherent.
        for snippet in (
            "S+2, KV axis",
            "exact multi-depth cache",
            "Terminal hidden-state cache",
            r"\alpha_{\rm kv}",
            "Prior Diagnostic Evidence",
            "16x1 Dense-MoE",
        ):
            self.assertIn(snippet, text)
        self.assertIn("quality gap", lowered)
        self.assertNotIn("0.9--0.99", text)
        self.assertNotIn("0.9 to 0.99", text)

    def test_opg_doc_models_readout_is_zk_to_mos_not_halting(self) -> None:
        r"""\S{}Models readout describes z_K -> MoS, with no halting readout.

        The halting-gate readout (eq:halting-readout) was P1 measurement
        machinery; the retained M0 readout feeds the final recurrent state
        z_K = 0.5(a_K + b_K) into a Mixture-of-Softmaxes head.
        """
        text = OPG_DOC.read_text()
        lowered = text.lower()
        # No halting readout in the core model description.
        self.assertNotIn("halting readout", lowered)
        self.assertNotIn("eq:halting-readout", text)
        self.assertNotIn(r"g_k=\sigma(w^\top z_k)", text)
        # z_K -> MoS readout is described: z_K = 0.5(a_K + b_K) fed to a MoS head.
        self.assertIn("z_K", text)
        self.assertRegex(text, r"(?:0\.5|\\tfrac12|\\frac12)\s*\\?!?\s*\\?left?\(?a_K\+b_K")
        self.assertIn("mixture-of-softmaxes", lowered)

    def test_opg_doc_has_related_work(self) -> None:
        text = OPG_DOC.read_text()
        self.assertIn(r"\section{Related Work}", text)
        for cite in ("revffn", "remoe", "isodepth", "deepseekv2", "mos"):
            self.assertIn(r"\bibitem{" + cite + "}", text)

    def test_opg_doc_demotes_internal_effective_depth_metrics(self) -> None:
        text = OPG_DOC.read_text()
        lowered = text.lower()
        for snippet in (
            "Recurrence-equivalence exponent",
            r"\mathrm{WFIR}",
            r"\mathrm{AEBR}",
            "ED_update",
            "ED_logit",
            "route-depth NMI",
        ):
            self.assertIn(snippet, text)
        self.assertIn("diagnostic", lowered)
        self.assertIn("not p1 gates", lowered)

    def test_operating_docs_reject_lyapunov_pressure(self) -> None:
        for path in (
            CLAUDE,
            ROOT / "docs" / "adr" / "0001-finite-horizon-opg-main-path.md",
            ROOT / "docs" / "adr" / "0002-final-minimal-p1-design.md",
        ):
            text = path.read_text()
            lowered = text.lower()
            self.assertIn("lyapunov_coef", text, str(path))
            self.assertRegex(lowered, r"reject|exclude|pressure", str(path))
            self.assertIn("diagnostic", lowered, str(path))

    def test_adr0002_interface_matches_harness_variants(self) -> None:
        """ADR 0002's Interface must list exactly the p1_synthetic --variant surface.

        Operationalizes the 'ADR interface <-> code consistency' directive: the
        decision record's interface cannot silently drift from the implementation.
        """
        import re

        adr = (ROOT / "docs" / "adr" / "0002-final-minimal-p1-design.md").read_text()
        src = (ROOT / "experiments" / "p1_synthetic.py").read_text()
        m = re.search(r'"--variant",\s*choices=\[([^\]]*)\]', src)
        assert m is not None, "could not locate --variant choices in p1_synthetic.py"
        variants = re.findall(r'"([a-z0-9_]+)"', m.group(1))
        self.assertEqual(set(variants), {"control", "m0", "mclk"})
        for v in variants:
            self.assertIn(v, adr, f"ADR 0002 Interface omits harness variant {v!r}")

    def test_context_names_final_minimal_p1_without_reactivating_legacy_consistency(self) -> None:
        text = (ROOT / "CONTEXT.md").read_text()
        for term in (
            "Final Minimal P1 Design",
            "P1 Synthetic Harness",
            "P1 Depth Utility",
            "Depth-Hard Synthetic",
            "Positive Control",
            "Clock Diagnostic",
            "Halting Readout",
            "Fixed-Point Diagnostics",
            "Terminal Hidden-State Cache",
            "Exact Multi-Depth Cache",
            "Expert Layout",
            "Expert Slot Order",
        ):
            self.assertIn(term, text)
        self.assertIn("Legacy fallback pressure", text)
        self.assertIn("startup validation rejects attempts to re-enable prefix anchors", text)

    def test_p1_synthetic_harness_exposes_only_core_p1_surface(self) -> None:
        text = P1_SYNTHETIC.read_text()
        pipeline = P1_SYNTHETIC_PIPELINE.read_text()
        for snippet in (
            '"control"',
            '"m0"',
            '"mclk"',
            "compose_s5_sequence",
            "reconstruction_error",
            "G_nll",
            "G_nll_ci_low",
            "G_nll_ci_high",
            "NDR_epsilon",
        ):
            self.assertIn(snippet, text)
        for removed in (
            '"moe"',
            '"static_moe"',
            "SoftMoEFFN",
            "cache_diagnostics",
            "_quantize_linear_int6_",
            "mechanism_diagnostics",
            "_route_depth_nmi",
            "_effective_rank",
            "HardGain",
        ):
            self.assertNotIn(removed, text)
        for snippet in (
            "for variant in control m0 mclk",
            'run_variant "${variant}"',
        ):
            self.assertIn(snippet, pipeline)
        for snippet in (
            'SEQ_LEN="${SEQ_LEN:-16}"',
            'TRAIN_DEPTHS="${TRAIN_DEPTHS:-16,32,64}"',
            'PAIRS="${PAIRS:-16,64,8,32}"',
            'NPROC="${NPROC:-2}"',
        ):
            self.assertIn(snippet, pipeline)
        for removed in ("STAGES=", "NUM_EXPERTS", "ROUTER_TOP_R", "MOE_BALANCE_COEF", "emit-cache", "eval-int6"):
            self.assertNotIn(removed, pipeline)
        self.assertIn("experiments/p1_synthetic.py", pipeline)
        self.assertIn("torchrun --standalone --nproc_per_node=2", CLAUDE.read_text())

    def test_feedback_stage_diagnostics_are_separate_from_p1_core(self) -> None:
        text = P1_FEEDBACK_STAGES.read_text()
        pipeline = P1_FEEDBACK_PIPELINE.read_text()
        for snippet in (
            "SoftMoEFFN",
            '"s1"',
            '"s2"',
            '"s3"',
            '"moe"',
            '"static_moe"',
            "mechanism_diagnostics",
            "_route_depth_nmi",
            "_effective_rank",
            "cache_diagnostics",
            "hidden_state_proxy_not_autoregressive_kv",
            "_quantize_linear_int6_",
            "quality_gap_status",
            "not_tested_requires_autoregressive_terminal_context_attention",
        ):
            self.assertIn(snippet, text)
        for snippet in (
            "experiments/p1_synthetic.py",
            "experiments/p1_feedback_stages.py",
            "run_stage_variant s1 m0",
            "run_stage_variant s1 moe",
            "run_stage_variant s1 static_moe",
            "run_stage_variant s2 m0",
            "run_stage_variant s3 m0",
            'NUM_EXPERTS="${NUM_EXPERTS:-4}"',
            'ROUTER_TOP_R="${ROUTER_TOP_R:-0}"',
        ):
            self.assertIn(snippet, pipeline)
        self.assertNotIn("dirichlet_ucb", text.lower())
        self.assertNotIn("ucb_beta", text.lower())

    def test_expert_layout_docs_use_explicit_slot_api(self) -> None:
        train = TRAIN_GPT.read_text()
        context = (ROOT / "CONTEXT.md").read_text()
        for snippet in (
            "experts_per_slot",
            "expert_slots",
            "expert_slot_order",
        ):
            self.assertIn(snippet, train)
        self.assertIn("expressiveness/throughput", context)
        self.assertIn("P3 MoE Basis", context)
        self.assertNotIn("use_chained_routing", train)
        self.assertNotIn("chained_stages_preset", train)

    def test_experiment_docs_are_canonical(self) -> None:
        # Canonical ACTIVE experiment docs. The superseded research history
        # (hypotheses_archive.md, iter103_chained_routing_plan.md) was archived
        # to legacy/docs/ in the 2026-06-06 repo reorg; see ADR 0003.
        expected_docs = [
            "README.md",
            "hypotheses.md",
            "recurrent_depth_baselines.md",
        ]
        for name in expected_docs:
            doc_path = EXPERIMENT_DOCS / name
            self.assertTrue(doc_path.exists(), name)
            # File-existence-only is too weak — an accidentally truncated
            # doc would pass. Require content sanity (≥200 chars of body).
            body = doc_path.read_text().strip()
            self.assertGreaterEqual(
                len(body), 200,
                f"{name} too short ({len(body)} chars) — likely truncated",
            )
        # The archived research history must live under legacy/docs/, not the
        # active experiments/docs/ tree.
        for name in ["hypotheses_archive.md", "iter103_chained_routing_plan.md"]:
            self.assertTrue(
                (LEGACY_DOCS / name).exists(),
                f"archived doc missing from legacy/docs/: {name}",
            )
            self.assertFalse(
                (EXPERIMENT_DOCS / name).exists(),
                f"archived doc still in active experiments/docs/: {name}",
            )
        for old_path in [
            ROOT / "experiments" / "hypotheses.md",
            ROOT / "experiments" / "hypotheses_archive.md",
            ROOT / "experiments" / "iter103_chained_routing_plan.md",
        ]:
            self.assertFalse(old_path.exists(), f"stale root-level doc remains: {old_path}")

        hyp = (EXPERIMENT_DOCS / "hypotheses.md").read_text()
        for snippet in (
            "final-minimal P1 pipeline",
            "RUN_TAG=gpu67_s0_pruned_i1000",
            "Legacy Dense-MoE Snapshot (Superseded)",
            "Legacy Remaining Queue (Superseded)",
            "Legacy Run Macros (Superseded)",
        ):
            self.assertIn(snippet, hyp)
        # Require "removed" to appear within a same-line "pure IFT" window
        # so an unrelated future use of the word "removed" elsewhere in the
        # doc cannot satisfy the IFT-refutation assertion. Single-line +
        # 200-char span is tight enough to survive minor reformatting but
        # not loose enough to span paragraphs.
        self.assertRegex(hyp, r"(?i)pure IFT[^\n]{0,200}remov")

    def test_iteration_workflow_requires_comprehensive_docs_review(self) -> None:
        expected = "review `experiments/docs/` comprehensively"
        self.assertIn(expected, CLAUDE.read_text().lower())
        self.assertIn(expected, EXPERIENCE.read_text().lower())
        self.assertIn(expected, (EXPERIMENT_DOCS / "README.md").read_text().lower())

    def test_router_dirichlet_diag_registry_in_sync(self) -> None:
        """plot_metrics.py mirrors train_gpt.py's Dirichlet-diag registry.

        The two registries are kept in lockstep so adding a new diagnostic
        is a one-line change in train_gpt.py — the parser does not need
        to be edited unless the *order* changes. Audit gate:
        CLAUDE.md "Sibling-fanout DRY gate".
        """
        import sys
        sys.path.insert(0, str(ROOT))
        from legacy.train_gpt_rich import (
            ROUTER_DIRICHLET_BETA_TERM,
            ROUTER_DIRICHLET_DIAG_TERMS,
        )
        from experiments.plot_metrics import ROUTER_DIRICHLET_DIAG_FIELDS
        canonical = tuple(name for name, _, _ in ROUTER_DIRICHLET_DIAG_TERMS) + (
            ROUTER_DIRICHLET_BETA_TERM[0],
        )
        self.assertEqual(canonical, ROUTER_DIRICHLET_DIAG_FIELDS)

    def test_no_stale_root_experiment_doc_paths(self) -> None:
        stale_paths = [
            "experiments/" + "hypotheses.md",
            "experiments/" + "hypotheses_archive.md",
            "experiments/" + "iter103_chained_routing_plan.md",
            "../" + "iter103_chained_routing_plan.md",
        ]
        checked_files = [
            TRAIN_GPT,
            CLAUDE,
            EXPERIENCE,
            OPG_DOC,
            ROOT / "legacy" / "components" / "README.md",
            EXPERIMENT_DOCS / "README.md",
            EXPERIMENT_DOCS / "hypotheses.md",
            LEGACY_DOCS / "hypotheses_archive.md",
            LEGACY_DOCS / "iter103_chained_routing_plan.md",
        ]
        for path in checked_files:
            text = path.read_text()
            for stale in stale_paths:
                self.assertNotIn(stale, text, f"{path} still references {stale}")

    def test_update_results_log_summary_tolerates_missing_val_bpb(self) -> None:
        text = UPDATE_RESULTS.read_text()
        bpb_lines = [line for line in text.splitlines() if "val_bpb:\\K" in line]
        self.assertEqual(len(bpb_lines), 1)
        self.assertIn("|| true", bpb_lines[0])

    def test_training_log_emits_auxiliary_loss_components(self) -> None:
        text = TRAIN_GPT.read_text()
        required_fields = [
            "router_cv_loss:",
            "router_pertoken_entropy_loss:",
            "mos_cv_loss:",
            "expert_diversity_loss:",
            "mos_diversity_loss:",
            "router_reg_loss:",
            "router_pertoken_entropy_coef_eff:",
            "expert_diversity_coef_eff:",
            "mos_diversity_coef_eff:",
            "scale_hinge_loss:",
        ]
        for field in required_fields:
            self.assertIn(field, text)
        self.assertIn("router_reg_loss:{_log_tensor_attr('_router_reg_loss_t')", text)

    def test_training_log_emits_parcae_diagnostics(self) -> None:
        text = TRAIN_GPT.read_text()
        required_fields = [
            "parcae_a_bar_min:",
            "parcae_a_bar_mean:",
            "parcae_a_bar_max:",
            "parcae_a_bar_core_max:",
            "parcae_beta_mean:",
            "parcae_beta_max:",
            "parcae_b_bar_mean:",
            "parcae_b_bar_max:",
            "parcae_delta_mean:",
            "parcae_delta_max:",
            "parcae_recon_amp_log10:",
        ]
        for field in required_fields:
            self.assertIn(field, text)
        self.assertIn("def parcae_diagnostics", text)

    def test_update_results_log_summary_tolerates_missing_auxiliary_fields(self) -> None:
        text = UPDATE_RESULTS.read_text()
        router_reg_lines = [line for line in text.splitlines() if "router_reg_loss:\\K" in line]
        self.assertEqual(len(router_reg_lines), 1)
        self.assertIn("|| true", router_reg_lines[0])
        self.assertIn("extract_aux_terms", text)
        for key in [
            "router_cv_loss",
            "router_pertoken_entropy_loss",
            "mos_cv_loss",
            "expert_diversity_loss",
            "mos_diversity_loss",
        ]:
            self.assertIn(key, text)

    def test_update_results_log_summary_tolerates_missing_parcae_fields(self) -> None:
        text = UPDATE_RESULTS.read_text()
        self.assertIn("extract_parcae_state", text)
        self.assertIn("parcae=${parcae_state:-?}", text)
        for key in [
            "parcae_a_bar_min",
            "parcae_a_bar_mean",
            "parcae_a_bar_max",
            "parcae_a_bar_core_max",
            "parcae_beta_mean",
            "parcae_beta_max",
            "parcae_b_bar_mean",
            "parcae_b_bar_max",
            "parcae_delta_mean",
            "parcae_delta_max",
            "parcae_recon_amp_log10",
        ]:
            self.assertIn(key, text)


if __name__ == "__main__":
    unittest.main()
