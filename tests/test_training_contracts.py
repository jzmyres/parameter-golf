import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_GPT = ROOT / "train_gpt.py"
UPDATE_RESULTS = ROOT / "experiments" / "update_results.sh"
OPG_DOC = ROOT / "opg_doc.tex"
CLAUDE = ROOT / "CLAUDE.md"
EXPERIENCE = ROOT / "EXPERIENCE.md"
EXPERIMENT_DOCS = ROOT / "experiments" / "docs"


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

    def test_opg_doc_tracks_active_entrypoint_and_rescue_defaults(self) -> None:
        # Whitespace-tolerant: any number of spaces/tabs between `\texttt{...}`,
        # the `&` separator, and the value. Matches the actual table column
        # alignment without breaking on harmless reformatting (one space vs
        # tab vs aligned-block) — only the semantic key/value pairing is
        # enforced.
        text = OPG_DOC.read_text()
        self.assertIn(r"\texttt{train\_gpt\_mlx.py} script is an Apple Silicon/MLX starter path", text)
        for pattern in (
            r"\\texttt\{deq\\_k\\_jitter\\_set\}\s*&\s*\$\(16,24,32,64,96,128\)\$",
            r"\\texttt\{config\\_profile\}\s*&\s*fast\\_default",
            r"\\texttt\{eval\\_profile\}\s*&\s*diagnostic",
            r"\\texttt\{diagnostic\\_gate\\_policy\}\s*&\s*advisory",
            r"\\texttt\{router\\_ema\\_alive\\_coef\}\s*&\s*0\.02",
            r"\\texttt\{expert\\_diversity\\_kind\}\s*&\s*cosine\s+max-pair",
        ):
            self.assertRegex(text, pattern)
        self.assertIn("Root-cause fix policy", text)
        self.assertIn("generic router usage controller", text)
        self.assertIn("transition-map", text)
        for snippet in (
            r"\paragraph{Profiles.}",
            r"\texttt{score\_iter152}",
            r"\texttt{eval\_profile=submission}",
            r"\paragraph{Feature and capability defaults.}",
            r"\texttt{use\_grouped\_artifact\_compression} & false & optional artifact-effect path",
            r"\texttt{use\_gptq}, \texttt{use\_lqer} & false, false & rejected scaffold",
            r"\texttt{use\_caseops} & false & rejected scaffold",
            r"\texttt{use\_sparse\_dispatch} & false & rejected:",
            r"\texttt{encode\_scored\_artifact}",
            r"\texttt{diagnostic} profile evaluates",
            r"\texttt{submission} profile evaluates",
            r"\texttt{debug} profile evaluates only",
            r"\texttt{score\_valid}",
            r"\texttt{health\_valid}",
            r"\texttt{diagnostic\_gate\_policy=hard}",
        ):
            self.assertIn(snippet, text)

    def test_experiment_docs_are_canonical(self) -> None:
        expected_docs = [
            "README.md",
            "hypotheses.md",
            "hypotheses_archive.md",
            "iter103_chained_routing_plan.md",
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
        for old_path in [
            ROOT / "experiments" / "hypotheses.md",
            ROOT / "experiments" / "hypotheses_archive.md",
            ROOT / "experiments" / "iter103_chained_routing_plan.md",
        ]:
            self.assertFalse(old_path.exists(), f"stale root-level doc remains: {old_path}")

        hyp = (EXPERIMENT_DOCS / "hypotheses.md").read_text()
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
        from train_gpt import (
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
            ROOT / "experiments" / "components" / "README.md",
            EXPERIMENT_DOCS / "README.md",
            EXPERIMENT_DOCS / "hypotheses.md",
            EXPERIMENT_DOCS / "hypotheses_archive.md",
            EXPERIMENT_DOCS / "iter103_chained_routing_plan.md",
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
            "consistency_anchor_loss:",
            "consistency_ext_loss:",
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
