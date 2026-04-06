"""Plot detailed training metrics comparing baseline vs current experiment.

Shows full training curves for ALL diagnostic metrics:
- Row 1: Train Loss, Val BPB, Step Avg (ms)
- Row 2: NTP Loss, CTP Loss, Pre-clip Grad Norm
- Row 3: DEQ Residual, DEQ Iter Conv (absolute), DEQ Iter Convergence (relative)
- Row 4: Expert Usage (min per component), Expert Entropy, Expert Orthogonality
- Row 5: Expert Balance CV (per component), DEQ Recon Error, Final Post-Quant Val BPB
- Row 6: GG by DEQ iter, Summary, (spare)

All subplots use consistent colors: blue for Baseline, orange for Current.
Components (mlp/attn/mos_ctp/mos_ntp) are encoded with line styles.
"""
import re
import sys
import math
from pathlib import Path

# Consistent colors: blue for Baseline, orange for Current
COLOR_BASELINE = "#1f77b4"  # matplotlib default blue
COLOR_CURRENT = "#ff7f0e"   # matplotlib default orange

# Component line styles (primary encoding)
COMP_LINESTYLES = {
    "transformer_block": "-",
    "mos_ctp": "--",
    "mos_ntp": ":",
}

COMP_SCATTER_SIZE = 72

_FLOAT = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"


def parse_log(logpath: str) -> dict:
    """Parse training log for all metrics."""
    lines = Path(logpath).read_text().split("\n")
    def _new_data() -> dict:
        return {
        "run_id": None,
        "config": {},
        "config_line": None,
        "train_batch_tokens": None,
        "train_steps": [], "train_loss": [], "ntp_loss": [], "ctp_loss": [],
        "grad_norm": [],
        "step_avg_ms": [], "train_time_ms": [],
        "val_steps": [], "val_loss": [], "val_bpb": [],
        "val_train_time_ms": [], "val_step_avg_ms": [],
        # Validation-time diagnostics (sparse unless VAL_LOSS_EVERY is small)
        "deq_residual": [], "deq_recon": [], "deq_iter_conv": [], "deq_iter_conv_rel": [],
        "gg_iter": [],
        # Combined expert metrics (backward compat)
        "expert_usage": [], "expert_entropy": [], "expert_sparsity": [], "expert_ortho": [],
        # Block-level expert orthogonality (explicit key; `expert_ortho` is kept for compat)
        "block_ortho": [],
        # Per-component: usage (list of lists), entropy, cv
        **{f"{p}_{s}": [] for p in ("mlp", "attn", "mos_ctp", "mos_ntp")
           for s in ("usage", "entropy", "cv")},
        # Block-level routing metrics (preferred for block-level MoE plotting).
        "block_usage": [], "block_entropy": [], "block_cv": [],
        # Orthogonality (expert_ortho is block-level weighted expert contribution orthogonality)
        "mos_ctp_ortho": [], "mos_ntp_ortho": [], "mos_ortho": [],
        # Train-time diagnostics (dense, logged alongside train_loss when enabled)
        "deq_residual_train": [], "deq_recon_train": [], "deq_iter_conv_train": [], "deq_iter_conv_rel_train": [],
        "gg_iter_train": [],
        **{f"{p}_{s}_train": [] for p in ("mlp", "attn", "mos_ctp", "mos_ntp")
           for s in ("usage", "entropy", "cv")},
        "block_usage_train": [], "block_entropy_train": [], "block_cv_train": [],
        "expert_ortho_train": [],
        "block_ortho_train": [],
        "mos_ctp_ortho_train": [], "mos_ntp_ortho_train": [],
        # (no weight-space orthogonality keys; enforce/plot output-space only)
        # Final post-quant scoring metric (what the submission is scored on)
        "final_postquant_val_loss": None,
        "final_postquant_val_bpb": None,
        }

    # NOTE: logs may accidentally contain multiple runs concatenated together (e.g. reused run_id).
    # Plotting must be run-session aware; keep only the most recent run.
    data = _new_data()
    last_step_seen: int | None = None

    for line in lines:
        m = re.search(r"^run_id:([\\w\\-\\.]+)", line)
        if m:
            data["run_id"] = m.group(1)

        m = re.search(r"^config:\\s*(.*)$", line)
        if m:
            data["config_line"] = m.group(1).strip()
            cfg: dict[str, str] = {}
            for part in data["config_line"].split():
                if "=" not in part:
                    continue
                k, v = part.split("=", 1)
                cfg[k.strip()] = v.strip()
            data["config"] = cfg

        m = re.search(r"^train_batch_tokens:(\d+)", line)
        if m:
            # Start of a new run: reset if we already parsed any steps/metrics.
            if data["train_steps"] or data["val_steps"] or data.get("final_postquant_val_bpb") is not None:
                data = _new_data()
                last_step_seen = None
            if data["train_batch_tokens"] is None:
                data["train_batch_tokens"] = int(m.group(1))

        # Training steps
        m = re.search(rf"^step:(\d+)/\d+ train_loss:{_FLOAT}.*train_time:{_FLOAT}ms step_avg:{_FLOAT}ms", line)
        if m:
            step_i = int(m.group(1))
            if last_step_seen is not None and step_i < last_step_seen:
                data = _new_data()
                last_step_seen = None
            last_step_seen = step_i
            data["train_steps"].append(step_i)
            data["train_loss"].append(float(m.group(2)))
            data["train_time_ms"].append(float(m.group(3)))
            data["step_avg_ms"].append(float(m.group(4)))
            # Parse NTP and CTP losses from training lines
            m_ntp = re.search(rf"ntp_loss:{_FLOAT}", line)
            data["ntp_loss"].append(float(m_ntp.group(1)) if m_ntp else math.nan)
            m_ctp = re.search(rf"ctp_loss:{_FLOAT}", line)
            data["ctp_loss"].append(float(m_ctp.group(1)) if m_ctp else math.nan)
            # Parse pre-clip gradient norm
            m_gn = re.search(rf"grad_norm:{_FLOAT}", line)
            data["grad_norm"].append(float(m_gn.group(1)) if m_gn else math.nan)

            # Train-time DEQ + expert diagnostics (optional). Missing values become NaN.
            for key, pat in [
                ("deq_residual_train", rf"deq_residual:{_FLOAT}"),
                ("deq_recon_train", rf"deq_recon_err:{_FLOAT}"),
                ("deq_iter_conv_train", rf"deq_iter_conv:{_FLOAT}"),
                ("deq_iter_conv_rel_train", rf"deq_iter_conv_rel:{_FLOAT}"),
                ("mlp_entropy_train", rf"mlp_entropy:{_FLOAT}"),
                ("attn_entropy_train", rf"attn_entropy:{_FLOAT}"),
                ("block_ortho_train", rf"block_ortho:{_FLOAT}"),
                ("expert_ortho_train", rf"expert_ortho:{_FLOAT}"),
                ("mos_ctp_entropy_train", rf"mos_ctp_entropy:{_FLOAT}"),
                ("mos_ntp_entropy_train", rf"mos_ntp_entropy:{_FLOAT}"),
                ("block_entropy_train", rf"block_entropy:{_FLOAT}"),
                ("mlp_cv_train", rf"mlp_cv:{_FLOAT}"),
                ("attn_cv_train", rf"attn_cv:{_FLOAT}"),
                ("mos_ctp_cv_train", rf"mos_ctp_cv:{_FLOAT}"),
                ("mos_ntp_cv_train", rf"mos_ntp_cv:{_FLOAT}"),
                ("block_cv_train", rf"block_cv:{_FLOAT}"),
                ("mos_ctp_ortho_train", rf"mos_ctp_ortho:{_FLOAT}"),
                ("mos_ntp_ortho_train", rf"mos_ntp_ortho:{_FLOAT}"),
            ]:
                m2 = re.search(pat, line)
                data[key].append(float(m2.group(1)) if m2 else math.nan)
            for prefix in ("mlp", "attn", "mos_ctp", "mos_ntp"):
                m_u = re.search(rf"{prefix}_usage:\[([\d.,\s]+)\]", line)
                data[f"{prefix}_usage_train"].append(
                    [float(v.strip()) for v in m_u.group(1).split(",") if v.strip()] if m_u else []
                )
            m_bu = re.search(r"block_usage:\[([\d.,\s]+)\]", line)
            data["block_usage_train"].append(
                [float(v.strip()) for v in m_bu.group(1).split(",") if v.strip()] if m_bu else []
            )
            m_gg = re.search(r"gg_iter:\[([\d.,\s]+)\]", line)
            data["gg_iter_train"].append(
                [float(v.strip()) for v in m_gg.group(1).split(",") if v.strip()] if m_gg else []
            )

        # Final post-quant metric (exact, if available)
        # Example:
        # final_int6_zstd_roundtrip_exact val_loss:2.53669845 val_bpb:1.50237571
        m = re.search(rf"^final_int6_\w+_roundtrip_exact val_loss:{_FLOAT} val_bpb:{_FLOAT}", line)
        if m:
            data["final_postquant_val_loss"] = float(m.group(1))
            data["final_postquant_val_bpb"] = float(m.group(2))

        # Validation steps
        m = re.search(rf"^step:(\d+)/\d+ val_loss:{_FLOAT} val_bpb:{_FLOAT}", line)
        if m:
            step_i = int(m.group(1))
            if last_step_seen is not None and step_i < last_step_seen:
                data = _new_data()
                last_step_seen = None
            last_step_seen = step_i
            data["val_steps"].append(step_i)
            data["val_loss"].append(float(m.group(2)))
            data["val_bpb"].append(float(m.group(3)))
            # Total training time is printed on val lines; use it as authoritative for summary.
            m_tt = re.search(rf"train_time:{_FLOAT}ms", line)
            data["val_train_time_ms"].append(float(m_tt.group(1)) if m_tt else math.nan)
            m_sa = re.search(rf"step_avg:{_FLOAT}ms", line)
            data["val_step_avg_ms"].append(float(m_sa.group(1)) if m_sa else math.nan)
            # Parse individual DEQ/expert metrics
            for key, pat in [
                ("deq_residual", rf"deq_residual:{_FLOAT}"),
                ("deq_recon", rf"deq_recon_err:{_FLOAT}"),
                ("deq_iter_conv", rf"deq_iter_conv:{_FLOAT}"),
                ("deq_iter_conv_rel", rf"deq_iter_conv_rel:{_FLOAT}"),
                ("expert_entropy", rf"(?<!\w_)expert_entropy:{_FLOAT}"),
                ("expert_sparsity", rf"(?<!\w_)expert_sparsity:{_FLOAT}"),
                ("block_ortho", rf"block_ortho:{_FLOAT}"),
                ("expert_ortho", rf"expert_ortho:{_FLOAT}"),
            ]:
                m2 = re.search(pat, line)
                data[key].append(float(m2.group(1)) if m2 else math.nan)
            # Combined expert usage (backward compat)
            m2 = re.search(r"(?<!\w_)expert_usage:\[([\d.,\s]+)\]", line)
            if m2:
                usage = [float(v.strip()) for v in m2.group(1).split(",") if v.strip()]
                data["expert_usage"].append(usage)
            else:
                data["expert_usage"].append([])
            # Per-component expert usage + entropy + cv
            for prefix in ("mlp", "attn", "mos_ctp", "mos_ntp"):
                m_u = re.search(rf"{prefix}_usage:\[([\d.,\s]+)\]", line)
                data[f"{prefix}_usage"].append(
                    [float(v.strip()) for v in m_u.group(1).split(",") if v.strip()] if m_u else [])
                m_e = re.search(rf"{prefix}_entropy:{_FLOAT}", line)
                data[f"{prefix}_entropy"].append(float(m_e.group(1)) if m_e else math.nan)
                m_cv = re.search(rf"{prefix}_cv:{_FLOAT}", line)
                data[f"{prefix}_cv"].append(float(m_cv.group(1)) if m_cv else math.nan)
            # Block-level router stats
            m_bu = re.search(r"block_usage:\[([\d.,\s]+)\]", line)
            data["block_usage"].append(
                [float(v.strip()) for v in m_bu.group(1).split(",") if v.strip()] if m_bu else []
            )
            m_be = re.search(rf"block_entropy:{_FLOAT}", line)
            data["block_entropy"].append(float(m_be.group(1)) if m_be else math.nan)
            m_bcv = re.search(rf"block_cv:{_FLOAT}", line)
            data["block_cv"].append(float(m_bcv.group(1)) if m_bcv else math.nan)
            # Orthogonality (MoS head)
            for comp in ("mos", "mos_ctp", "mos_ntp"):
                m_o = re.search(rf"{comp}_ortho:{_FLOAT}", line)
                data[f"{comp}_ortho"].append(float(m_o.group(1)) if m_o else math.nan)
            # No balance-loss keys are logged; use *_cv fields for balance diagnostics.
            m_gg = re.search(r"gg_iter:\[([\d.,\s]+)\]", line)
            data["gg_iter"].append(
                [float(v.strip()) for v in m_gg.group(1).split(",") if v.strip()] if m_gg else []
            )

    return data


def usage_min_series(data: dict, key: str) -> list[float]:
    """Return min expert usage per step for a usage list-of-lists key."""
    out: list[float] = []
    for u in data.get(key, []):
        out.append(min(u) if u else math.nan)
    return out


def _is_finite(x: float) -> bool:
    return x is not None and isinstance(x, (int, float)) and not math.isnan(float(x)) and math.isfinite(float(x))


def _filter_finite(steps: list[int], values: list[float]) -> tuple[list[int], list[float]]:
    if not steps or not values:
        return [], []
    n = min(len(steps), len(values))
    xs: list[int] = []
    ys: list[float] = []
    for i in range(n):
        v = values[i]
        if _is_finite(v):
            xs.append(steps[i])
            ys.append(float(v))
    return xs, ys


def _has_any_finite(values: list[float]) -> bool:
    return any(_is_finite(v) for v in values or [])

def _series_equal(a: list[float] | None, b: list[float] | None, *, eps: float = 1e-12) -> bool:
    """Approx equality for two numeric series, ignoring NaNs."""
    if a is None or b is None:
        return False
    n = min(len(a), len(b))
    if n == 0:
        return False
    any_compared = False
    for i in range(n):
        av = a[i]
        bv = b[i]
        if not (_is_finite(av) and _is_finite(bv)):
            continue
        any_compared = True
        if abs(float(av) - float(bv)) > eps:
            return False
    return any_compared


def _plot_line(ax, b, c, b_key, c_key, b_steps, c_steps, title, ylabel=None):
    """Plot two line series on the same axis with consistent colors."""
    plotted = False
    bx, by = _filter_finite(b.get(b_steps, []), b.get(b_key, []))
    cx, cy = _filter_finite(c.get(c_steps, []), c.get(c_key, []))
    if bx and by:
        if len(by) < 2:
            ax.scatter(bx, by, color=COLOR_BASELINE, alpha=0.85, label="Baseline", s=28)
            ax.text(bx[0], by[0], f"{by[0]:.4f}", fontsize=8, ha="left", va="bottom", color=COLOR_BASELINE)
        else:
            ax.plot(bx, by, color=COLOR_BASELINE, alpha=0.7, label="Baseline", linewidth=1.5)
        plotted = True
    if cx and cy:
        if len(cy) < 2:
            ax.scatter(cx, cy, color=COLOR_CURRENT, alpha=0.85, label="Current", s=28)
            ax.text(cx[0], cy[0], f"{cy[0]:.4f}", fontsize=8, ha="left", va="bottom", color=COLOR_CURRENT)
        else:
            ax.plot(cx, cy, color=COLOR_CURRENT, alpha=0.7, label="Current", linewidth=1.5)
        plotted = True
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])


def _plot_components(ax, b, c, steps_key, component_series, title, ylabel=None):
    """Plot multiple component series with Baseline=blue and Current=orange.

    Components are encoded via line styles for readability.
    """
    any_plotted = False
    for comp_label, b_values, c_values in component_series:
        linestyle = COMP_LINESTYLES.get(comp_label, "-")
        if b_values is not None and _has_any_finite(b_values):
            bx, by = _filter_finite(b.get(steps_key, []), b_values)
            if bx and by:
                if len(by) < 2:
                    ax.scatter(
                        bx,
                        by,
                        color=COLOR_BASELINE,
                        alpha=0.85,
                        s=COMP_SCATTER_SIZE,
                    )
                else:
                    ax.plot(
                        bx,
                        by,
                        color=COLOR_BASELINE,
                        linestyle=linestyle,
                        alpha=0.8,
                        linewidth=2.3,
                    )
                any_plotted = True
        if c_values is not None and _has_any_finite(c_values):
            cx, cy = _filter_finite(c.get(steps_key, []), c_values)
            if cx and cy:
                if len(cy) < 2:
                    ax.scatter(
                        cx,
                        cy,
                        color=COLOR_CURRENT,
                        alpha=0.85,
                        s=COMP_SCATTER_SIZE,
                    )
                else:
                    ax.plot(
                        cx,
                        cy,
                        color=COLOR_CURRENT,
                        linestyle=linestyle,
                        alpha=0.8,
                        linewidth=2.3,
                    )
                any_plotted = True

    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Step")
    if ylabel:
        ax.set_ylabel(ylabel)
    if not any_plotted:
        ax.text(0.5, 0.5, "Not logged", ha="center", va="center", fontsize=10, transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.grid(True, alpha=0.3)

    if any_plotted:
        try:
            from matplotlib.lines import Line2D
        except Exception:
            return
        # Two legends: colors (baseline/current) + line styles (components).
        color_handles = [
            Line2D([0], [0], color=COLOR_BASELINE, lw=2.5, label="Baseline"),
            Line2D([0], [0], color=COLOR_CURRENT, lw=2.5, label="Current"),
        ]
        style_handles = []
        for comp_label, _, _ in component_series:
            if comp_label not in COMP_LINESTYLES:
                continue
            style_handles.append(
                Line2D([0], [0], color="#333333", lw=2.5, linestyle=COMP_LINESTYLES[comp_label], label=comp_label)
            )
        if color_handles:
            leg1 = ax.legend(handles=color_handles, loc="upper left", fontsize=8, frameon=False)
            ax.add_artist(leg1)
        if style_handles:
            ax.legend(handles=style_handles, loc="upper right", fontsize=8, frameon=False)


def plot_comparison(baseline_log: str, current_log: str, outdir: str) -> bool:
    """Plot baseline vs current experiment comparison with full training curves.

    Returns True if a plot was generated, False if plotting dependencies are missing.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available", file=sys.stderr)
        return False

    b = parse_log(baseline_log)
    c = parse_log(current_log)

    def _prefer_train(train_key: str, val_key: str) -> tuple[str, str]:
        if _has_any_finite(b.get(train_key, [])) or _has_any_finite(c.get(train_key, [])):
            return "train_steps", train_key
        return "val_steps", val_key

    def _prefer_train_usage(train_key: str, val_key: str) -> tuple[str, str]:
        # usage keys are lists-of-lists; treat "logged" as "any non-empty entry"
        if any(len(u) for u in b.get(train_key, [])) or any(len(u) for u in c.get(train_key, [])):
            return "train_steps", train_key
        return "val_steps", val_key

    fig, axes = plt.subplots(6, 3, figsize=(18, 26))
    fig.suptitle("Baseline vs Current Experiment — Full Diagnostics", fontsize=16, fontweight="bold")

    # Global legend (colors = run, line style = component)
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color=COLOR_BASELINE, lw=2.2, label="Baseline"),
        Line2D([0], [0], color=COLOR_CURRENT, lw=2.2, label="Current"),
    ]
    for comp in ("transformer_block", "mos_ctp", "mos_ntp"):
        handles.append(
            Line2D(
                [0],
                [0],
                color="black",
                lw=2.0,
                linestyle=COMP_LINESTYLES.get(comp, "-"),
                label=comp,
            )
        )
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(top=0.93)

    # Row 1: Training metrics (total loss, val bpb, step avg)
    _plot_line(axes[0, 0], b, c, "train_loss", "train_loss", "train_steps", "train_steps", "Train Loss (Total)")
    _plot_line(axes[0, 1], b, c, "val_bpb", "val_bpb", "val_steps", "val_steps", "Val BPB")
    # Normalize step time to make runs with different batch sizes comparable.
    def _ms_per_mtok(d: dict) -> list[float]:
        tbt = d.get("train_batch_tokens", None)
        if not tbt:
            return list(d.get("step_avg_ms", []))
        return [ms * 1e6 / float(tbt) for ms in d.get("step_avg_ms", [])]

    b["step_ms_per_mtok"] = _ms_per_mtok(b)
    c["step_ms_per_mtok"] = _ms_per_mtok(c)
    _plot_line(
        axes[0, 2],
        b,
        c,
        "step_ms_per_mtok",
        "step_ms_per_mtok",
        "train_steps",
        "train_steps",
        "Step Avg (ms / 1M tok)",
    )

    # Row 2: NTP Loss, CTP Loss, Pre-clip Grad Norm
    _plot_line(axes[1, 0], b, c, "ntp_loss", "ntp_loss", "train_steps", "train_steps", "NTP Loss")
    _plot_line(axes[1, 1], b, c, "ctp_loss", "ctp_loss", "train_steps", "train_steps", "CTP Loss")
    _plot_line(axes[1, 2], b, c, "grad_norm", "grad_norm", "train_steps", "train_steps",
               "Pre-clip Grad Norm")

    # Row 3: DEQ diagnostics (prefer train-logged diagnostics for dense curves)
    steps_key, key = _prefer_train("deq_residual_train", "deq_residual")
    _plot_line(axes[2, 0], b, c, key, key, steps_key, steps_key, "DEQ Residual ||z - f(z)||")
    # (Requested swap sequence) DEQ iteration convergence (absolute) into Row 3 middle.
    steps_key, key = _prefer_train("deq_iter_conv_train", "deq_iter_conv")
    _plot_line(axes[2, 1], b, c, key, key, steps_key, steps_key, "DEQ Iter Conv (absolute)")
    steps_key, key = _prefer_train("deq_iter_conv_rel_train", "deq_iter_conv_rel")
    _plot_line(axes[2, 2], b, c, key, key, steps_key, steps_key, "DEQ Iter Conv (relative)")

    # Row 4: Expert diagnostics (usage, entropy, orthogonality). Prefer train-logged series for dense curves.
    # Usage: min share per expert group (Transformer block, MoS CTP, MoS NTP)
    ax_usage = axes[3, 0]
    use_train = any(
        _has_any_finite(usage_min_series(d, "block_usage_train"))
        or _has_any_finite(usage_min_series(d, "mos_ctp_usage_train"))
        or _has_any_finite(usage_min_series(d, "mos_ntp_usage_train"))
        for d in (b, c)
    )
    steps_key = "train_steps" if use_train else "val_steps"
    if use_train:
        usage_series = [
            ("transformer_block", usage_min_series(b, "block_usage_train"), usage_min_series(c, "block_usage_train")),
            ("mos_ctp", usage_min_series(b, "mos_ctp_usage_train"), usage_min_series(c, "mos_ctp_usage_train")),
            ("mos_ntp", usage_min_series(b, "mos_ntp_usage_train"), usage_min_series(c, "mos_ntp_usage_train")),
        ]
    else:
        usage_series = [
            ("transformer_block", usage_min_series(b, "block_usage"), usage_min_series(c, "block_usage")),
            ("mos_ctp", usage_min_series(b, "mos_ctp_usage"), usage_min_series(c, "mos_ctp_usage")),
            ("mos_ntp", usage_min_series(b, "mos_ntp_usage"), usage_min_series(c, "mos_ntp_usage")),
        ]
    _plot_components(
        ax_usage,
        b,
        c,
        steps_key,
        usage_series,
        "Expert Usage (min share by Group)",
        ylabel="Min usage fraction",
    )

    # Expert Entropy: per-group lines (Transformer block, MoS CTP, MoS NTP)
    ax_ent = axes[3, 1]
    use_train = any(
        _has_any_finite(d.get("block_entropy_train", [])) or _has_any_finite(d.get("mos_ctp_entropy_train", [])) or _has_any_finite(d.get("mos_ntp_entropy_train", []))
        for d in (b, c)
    )
    steps_key = "train_steps" if use_train else "val_steps"
    entropy_series = [
        ("transformer_block", b.get("block_entropy_train" if use_train else "block_entropy", []), c.get("block_entropy_train" if use_train else "block_entropy", [])),
        ("mos_ctp", b.get("mos_ctp_entropy_train" if use_train else "mos_ctp_entropy", []), c.get("mos_ctp_entropy_train" if use_train else "mos_ctp_entropy", [])),
        ("mos_ntp", b.get("mos_ntp_entropy_train" if use_train else "mos_ntp_entropy", []), c.get("mos_ntp_entropy_train" if use_train else "mos_ntp_entropy", [])),
    ]
    _plot_components(ax_ent, b, c, steps_key, entropy_series, "Expert Entropy (by Group)", ylabel="Entropy")

    # Orthogonality: block-level weighted experts + MoS head orthogonality
    ax_ortho = axes[3, 2]
    # Prefer train-logged series when any of the orthogonality metrics are present there.
    use_train = False
    for k in ("block_ortho_train", "expert_ortho_train", "mos_ctp_ortho_train", "mos_ntp_ortho_train"):
        if _has_any_finite(b.get(k, [])) or _has_any_finite(c.get(k, [])):
            use_train = True
            break
    steps_key = "train_steps" if use_train else "val_steps"
    expert_ortho_key = "block_ortho_train" if steps_key == "train_steps" else "block_ortho"
    ortho_series = []
    # Transformer block expert orthogonality (preferred explicit key; fall back to legacy `expert_ortho`).
    ortho_series.append((
        "transformer_block",
        b.get(expert_ortho_key, b.get("expert_ortho_train" if steps_key == "train_steps" else "expert_ortho", [])),
        c.get(expert_ortho_key, c.get("expert_ortho_train" if steps_key == "train_steps" else "expert_ortho", [])),
    ))
    ortho_series.append((
        "mos_ctp",
        b.get("mos_ctp_ortho_train" if steps_key == "train_steps" else "mos_ctp_ortho", []),
        c.get("mos_ctp_ortho_train" if steps_key == "train_steps" else "mos_ctp_ortho", []),
    ))
    ortho_series.append((
        "mos_ntp",
        b.get("mos_ntp_ortho_train" if steps_key == "train_steps" else "mos_ntp_ortho", []),
        c.get("mos_ntp_ortho_train" if steps_key == "train_steps" else "mos_ntp_ortho", []),
    ))
    _plot_components(
        ax_ortho,
        b,
        c,
        steps_key,
        ortho_series,
        "Orthogonality (max mean |cos| by Group)",
        ylabel="Max mean |cos|",
    )
    # Orthogonality is naturally bounded in [0, 1]; keep linear for interpretability.
    all_vals = []
    for _, bv, cv in ortho_series:
        all_vals.extend([v for v in (bv or []) if _is_finite(v)])
        all_vals.extend([v for v in (cv or []) if _is_finite(v)])
    hi = max(all_vals) if all_vals else 1.0
    ax_ortho.set_ylim(0.0, max(1.0, hi * 1.05))

    # Row 5: Balance diagnostics (CV), DEQ reconstruction error, scored metric
    ax_bal = axes[4, 0]
    use_train = any(
        _has_any_finite(d.get("block_cv_train", [])) or _has_any_finite(d.get("mos_ctp_cv_train", [])) or _has_any_finite(d.get("mos_ntp_cv_train", []))
        for d in (b, c)
    )
    steps_key = "train_steps" if use_train else "val_steps"
    bal_series = [
        ("transformer_block", b.get("block_cv_train" if use_train else "block_cv", []), c.get("block_cv_train" if use_train else "block_cv", [])),
        ("mos_ctp", b.get("mos_ctp_cv_train" if use_train else "mos_ctp_cv", []), c.get("mos_ctp_cv_train" if use_train else "mos_ctp_cv", [])),
        ("mos_ntp", b.get("mos_ntp_cv_train" if use_train else "mos_ntp_cv", []), c.get("mos_ntp_cv_train" if use_train else "mos_ntp_cv", [])),
    ]
    _plot_components(ax_bal, b, c, steps_key, bal_series, "Expert Balance CV (by Group)", ylabel="CV")

    # DEQ reconstruction error (prefer dense train-logged series when available)
    steps_key, key = _prefer_train("deq_recon_train", "deq_recon")
    _plot_line(axes[4, 1], b, c, key, key, steps_key, steps_key, "DEQ Reconstruction Error")

    # Pre vs post-quant val_bpb (post-quant is the scored metric)
    # (Requested swap) Put this bar chart in Row 6 left.
    ax_postq = axes[5, 0]
    ax_postq.set_title("Val BPB (Pre vs Post-Quant)", fontsize=11)
    b_pre = b["val_bpb"][-1] if b.get("val_bpb") else None
    c_pre = c["val_bpb"][-1] if c.get("val_bpb") else None
    b_post = b.get("final_postquant_val_bpb", None)
    c_post = c.get("final_postquant_val_bpb", None)

    width = 0.35
    x = [0.0, 1.0]
    # Matplotlib bar() can't handle None; use NaN for "missing".
    pre_vals = [float(b_pre) if b_pre is not None else math.nan, float(c_pre) if c_pre is not None else math.nan]
    post_vals = [float(b_post) if b_post is not None else math.nan, float(c_post) if c_post is not None else math.nan]
    pre_colors = [COLOR_BASELINE, COLOR_CURRENT]
    post_colors = [COLOR_BASELINE, COLOR_CURRENT]

    plotted_any = False
    if any(_is_finite(v) for v in pre_vals):
        ax_postq.bar([xi - width / 2 for xi in x], pre_vals, width=width, color=pre_colors, alpha=0.45, label="Pre-quant")
        plotted_any = True
    if any(_is_finite(v) for v in post_vals):
        ax_postq.bar([xi + width / 2 for xi in x], post_vals, width=width, color=post_colors, alpha=0.90, label="Post-quant")
        plotted_any = True

    if plotted_any:
        for xi, v in zip([x[0] - width / 2, x[1] - width / 2], pre_vals, strict=False):
            if _is_finite(v):
                ax_postq.annotate(
                    f"{v:.4f}",
                    xy=(xi, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )
        for xi, v in zip([x[0] + width / 2, x[1] + width / 2], post_vals, strict=False):
            if _is_finite(v):
                ax_postq.annotate(
                    f"{v:.4f}",
                    xy=(xi, v),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )
        ax_postq.set_xticks(x)
        ax_postq.set_xticklabels(["Baseline", "Current"])
        ax_postq.set_ylabel("val_bpb")
        ax_postq.legend(fontsize=8)
        ax_postq.grid(True, axis="y", alpha=0.3)
        vals = [v for v in pre_vals + post_vals if _is_finite(v)]
        if vals:
            lo = min(vals)
            hi = max(vals)
            pad = max((hi - lo) * 0.5, 0.002)
            ax_postq.set_ylim(lo - pad, hi + pad)
    else:
        ax_postq.text(
            0.5,
            0.5,
            "No val_bpb or post-quant metric found",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax_postq.transAxes,
        )
        ax_postq.set_xticks([])
        ax_postq.set_yticks([])

    # Row 6: Summary + spare
    def _prefer_train_listlist(train_key: str, val_key: str) -> tuple[str, str]:
        if any(len(v) for v in b.get(train_key, [])) or any(len(v) for v in c.get(train_key, [])):
            return "train_steps", train_key
        return "val_steps", val_key

    def _k_linestyle(ki: int):
        """Line style for GG-by-DEQ-iter curves.

        ki is 0-based; k=1 should be solid, and larger k should become progressively more dashed.
        """
        if ki <= 0:
            return "-"
        # k=2..12: increasingly "dashy" styles (shorter dashes / more frequent gaps).
        dash_table = [
            (0, (10, 3)),  # k=2
            (0, (9, 3)),   # k=3
            (0, (8, 3)),   # k=4
            (0, (7, 3)),   # k=5
            (0, (6, 3)),   # k=6
            (0, (5, 2)),   # k=7
            (0, (4, 2)),   # k=8
            (0, (3, 2)),   # k=9
            (0, (2, 2)),   # k=10
            (0, (2, 1)),   # k=11
            (0, (1, 1)),   # k=12
        ]
        idx = min(ki - 1, len(dash_table) - 1)
        return dash_table[idx]

    # (Requested swap) Put GG plot in Row 5 right.
    ax_gg = axes[4, 2]
    steps_key, key = _prefer_train_listlist("gg_iter_train", "gg_iter")
    b_iters = b.get(key, []) or []
    c_iters = c.get(key, []) or []
    k_plot = 0
    if b_iters:
        k_plot = max(k_plot, max((len(v) for v in b_iters), default=0))
    if c_iters:
        k_plot = max(k_plot, max((len(v) for v in c_iters), default=0))
    ax_gg.set_title("GG by DEQ Iter (avg across refinements)", fontsize=11)
    ax_gg.set_xlabel("Step")
    ax_gg.set_ylabel("gg")
    ax_gg.grid(True, alpha=0.3)
    if k_plot <= 0:
        ax_gg.text(0.5, 0.5, "Not logged", ha="center", va="center", fontsize=10, transform=ax_gg.transAxes)
        ax_gg.set_xticks([])
        ax_gg.set_yticks([])
    else:
        from matplotlib.lines import Line2D
        style_handles = []
        for ki in range(k_plot):
            style = _k_linestyle(ki)
            b_vals = [v[ki] if len(v) > ki else math.nan for v in b_iters]
            c_vals = [v[ki] if len(v) > ki else math.nan for v in c_iters]
            bx, by = _filter_finite(b.get(steps_key, []), b_vals)
            cx, cy = _filter_finite(c.get(steps_key, []), c_vals)
            if bx and by:
                ax_gg.plot(bx, by, color=COLOR_BASELINE, linestyle=style, alpha=0.75, linewidth=2.0)
            if cx and cy:
                ax_gg.plot(cx, cy, color=COLOR_CURRENT, linestyle=style, alpha=0.75, linewidth=2.0)
            style_handles.append(Line2D([0], [0], color="#333333", lw=2.2, linestyle=style, label=f"k={ki+1}"))
        ax_gg.set_ylim(0.0, 1.0)
        ax_gg.legend(handles=style_handles, loc="upper right", fontsize=8, frameon=False)

    axes[5, 1].axis("off")
    axes[5, 2].axis("off")

    summary_lines = []
    # Run metadata + config deltas (if logged)
    if b.get("run_id") or c.get("run_id"):
        summary_lines.append(f"Run ID:     {b.get('run_id') or '??'} -> {c.get('run_id') or '??'}")
    b_cfg = b.get("config") or {}
    c_cfg = c.get("config") or {}
    if b_cfg and c_cfg:
        shared = {k: b_cfg[k] for k in b_cfg.keys() & c_cfg.keys() if b_cfg.get(k) == c_cfg.get(k)}
        changed = {k: (b_cfg.get(k), c_cfg.get(k)) for k in (b_cfg.keys() | c_cfg.keys()) if b_cfg.get(k) != c_cfg.get(k)}
        if shared:
            items = " ".join(f"{k}={v}" for k, v in sorted(shared.items()))
            summary_lines.append(f"Shared:     {items}")
        if changed:
            items = " ".join(f"{k}:{bv}->{cv}" for k, (bv, cv) in sorted(changed.items()))
            summary_lines.append(f"Diff:       {items}")
    if b["val_bpb"] and c["val_bpb"]:
        delta = c["val_bpb"][-1] - b["val_bpb"][-1]
        summary_lines.append(f"Val BPB:    {b['val_bpb'][-1]:.4f} -> {c['val_bpb'][-1]:.4f} (d={delta:+.4f})")
    if b.get("final_postquant_val_bpb") is not None and c.get("final_postquant_val_bpb") is not None:
        d2 = c["final_postquant_val_bpb"] - b["final_postquant_val_bpb"]
        summary_lines.append(
            f"Post-Quant: {b['final_postquant_val_bpb']:.4f} -> {c['final_postquant_val_bpb']:.4f} (d={d2:+.4f})"
        )
    if b["train_steps"] and c["train_steps"]:
        summary_lines.append(f"Steps:      {b['train_steps'][-1]} vs {c['train_steps'][-1]}")
    # Prefer total training time from val lines (available even when train logs are sparse).
    def _last_finite(vals: list[float] | None) -> float:
        for v in reversed(vals or []):
            if _is_finite(v):
                return float(v)
        return math.nan

    b_time_ms = _last_finite(b.get("val_train_time_ms", []))
    c_time_ms = _last_finite(c.get("val_train_time_ms", []))
    if not _is_finite(b_time_ms):
        b_time_ms = _last_finite(b.get("train_time_ms", []))
    if not _is_finite(c_time_ms):
        c_time_ms = _last_finite(c.get("train_time_ms", []))
    if _is_finite(b_time_ms) and _is_finite(c_time_ms):
        summary_lines.append(f"Time (s):   {b_time_ms/1000.0:.1f} -> {c_time_ms/1000.0:.1f} (d={(c_time_ms-b_time_ms)/1000.0:+.1f})")
    if b["train_loss"] and c["train_loss"]:
        summary_lines.append(f"Train Loss: {b['train_loss'][-1]:.4f} vs {c['train_loss'][-1]:.4f}")
    if b["ntp_loss"] and c["ntp_loss"] and any(v > 0 for v in b["ntp_loss"]):
        summary_lines.append(f"NTP Loss:   {b['ntp_loss'][-1]:.4f} vs {c['ntp_loss'][-1]:.4f}")
    if b["ctp_loss"] and c["ctp_loss"] and any(v > 0 for v in b["ctp_loss"]):
        summary_lines.append(f"CTP Loss:   {b['ctp_loss'][-1]:.4f} vs {c['ctp_loss'][-1]:.4f}")
    if b["grad_norm"] and c["grad_norm"] and any(v > 0 for v in b["grad_norm"] + c["grad_norm"]):
        summary_lines.append(f"Grad Norm:  {b['grad_norm'][-1]:.4f} vs {c['grad_norm'][-1]:.4f}")
    if b["deq_residual"] and c["deq_residual"]:
        summary_lines.append(f"DEQ Res:    {b['deq_residual'][-1]:.0f} vs {c['deq_residual'][-1]:.0f}")
    if b["deq_recon"] and c["deq_recon"]:
        summary_lines.append(f"Recon Err:  {b['deq_recon'][-1]:.3e} vs {c['deq_recon'][-1]:.3e}")
    if b["deq_iter_conv"] and c["deq_iter_conv"]:
        summary_lines.append(f"Iter Conv:  {b['deq_iter_conv'][-1]:.1f} vs {c['deq_iter_conv'][-1]:.1f}")
    if b["expert_entropy"] and c["expert_entropy"]:
        summary_lines.append(f"Entropy:    {b['expert_entropy'][-1]:.4f} vs {c['expert_entropy'][-1]:.4f}")
    if b.get("expert_sparsity") and c.get("expert_sparsity") and b["expert_sparsity"] and c["expert_sparsity"]:
        summary_lines.append(f"Sparsity:   {b['expert_sparsity'][-1]:.4f} vs {c['expert_sparsity'][-1]:.4f}")
    if b["expert_ortho"] and c["expert_ortho"]:
        summary_lines.append(f"Ortho:      {b['expert_ortho'][-1]:.4f} vs {c['expert_ortho'][-1]:.4f}")
    if b.get("mos_ctp_ortho") and c.get("mos_ctp_ortho") and b["mos_ctp_ortho"] and c["mos_ctp_ortho"]:
        summary_lines.append(f"MoS CTP O:  {b['mos_ctp_ortho'][-1]:.4f} vs {c['mos_ctp_ortho'][-1]:.4f}")
    if b.get("mos_ntp_ortho") and c.get("mos_ntp_ortho") and b["mos_ntp_ortho"] and c["mos_ntp_ortho"]:
        summary_lines.append(f"MoS NTP O:  {b['mos_ntp_ortho'][-1]:.4f} vs {c['mos_ntp_ortho'][-1]:.4f}")
    summary = "\n".join(summary_lines)
    # (Requested swap) Put summary text into Row 6 middle.
    axes[5, 1].text(
        0.0,
        0.5,
        summary,
        fontsize=11,
        family="monospace",
        verticalalignment="center",
        transform=axes[5, 1].transAxes,
    )

    plt.tight_layout()
    plt.savefig(str(Path(outdir) / "metrics_comparison.png"), dpi=150)
    plt.close()
    print(f"Saved metrics_comparison.png")
    return True


if __name__ == "__main__":
    expdir = Path(__file__).resolve().parent
    logdir = expdir / "training_logs"
    baseline = logdir / "baseline.log"
    current = logdir / "current.log"

    if not baseline.exists():
        print(f"No baseline log at {baseline}")
        sys.exit(1)
    if not current.exists():
        print("No current log — using baseline for both")
        current = baseline

    ok = plot_comparison(str(baseline), str(current), str(expdir))
    if not ok:
        sys.exit(1)
