"""Optional experiment plot refresh hook used by ``train_gpt.py``.

This module intentionally imports plotting code lazily. Training and
submission runs must keep working when the ``experiments`` package or
matplotlib is absent.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import shutil
from pathlib import Path
from typing import Any


def _resolve_root(root: str | Path | None) -> Path:
    if root is not None:
        return Path(root).resolve()
    return Path(__file__).resolve().parent.parent


def _resolve_logfile(logfile: str | Path | None, root: Path) -> Path | None:
    if logfile is None:
        return None
    path = Path(logfile)
    if path.exists():
        return path
    root_path = root / path
    if root_path.exists():
        return root_path
    return None


def _import_module(name: str) -> Any | None:
    try:
        return importlib.import_module(name)
    except Exception:
        return None


def maybe_update_experiment_plots(
    logfile: str | Path | None,
    *,
    enabled: bool = True,
    root: str | Path | None = None,
) -> bool:
    """Refresh debug plots if plotting dependencies are importable.

    Returns ``True`` if at least one plotter reported success. All import and
    plotting failures are treated as debug-only no-ops.
    """
    if not enabled:
        return False

    project_root = _resolve_root(root)
    expdir = project_root / "experiments"
    logdir = expdir / "training_logs"
    current = logdir / "current.log"
    baseline = logdir / "baseline.log"

    try:
        logdir.mkdir(parents=True, exist_ok=True)
        source = _resolve_logfile(logfile, project_root)
        if source is not None:
            shutil.copyfile(source, current)
        if current.exists() and not baseline.exists():
            shutil.copyfile(current, baseline)
    except Exception:
        return False

    if not current.exists() or not baseline.exists():
        return False

    matplotlib = _import_module("matplotlib")
    if matplotlib is None:
        return False
    try:
        matplotlib.use("Agg", force=True)
    except Exception:
        return False

    plot_metrics = _import_module("experiments.plot_metrics")
    plot_eval_metrics = _import_module("experiments.plot_eval_metrics")
    plot_progress = _import_module("experiments.plot_progress")
    if plot_metrics is None and plot_eval_metrics is None and plot_progress is None:
        return False

    generated = False
    sink = io.StringIO()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            if plot_metrics is not None:
                generated = bool(plot_metrics.plot_comparison(str(baseline), str(current), str(expdir))) or generated
        except Exception:
            pass
        try:
            if plot_eval_metrics is not None:
                generated = bool(plot_eval_metrics.plot_eval_comparison(str(baseline), str(current), str(expdir))) or generated
        except Exception:
            pass
        try:
            if plot_progress is not None:
                progress_paths = (expdir / "progress.png", expdir / "progress_full.png")
                before = {
                    p: (p.stat().st_mtime_ns if p.exists() else None)
                    for p in progress_paths
                }
                plot_progress.main()
                generated = any(
                    p.exists() and p.stat().st_mtime_ns != before[p]
                    for p in progress_paths
                ) or generated
        except Exception:
            pass
    return generated
