import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.plotting_hook import maybe_update_experiment_plots


class TestPlottingHook(unittest.TestCase):
    def test_missing_plotting_imports_are_noop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logfile = root / "logs" / "run.txt"
            logfile.parent.mkdir()
            logfile.write_text("step:0/1 val_loss:1.0 val_bpb:1.0\n")

            with mock.patch(
                "experiments.plotting_hook.importlib.import_module",
                side_effect=ImportError("missing"),
            ):
                ok = maybe_update_experiment_plots(logfile, root=root)

            self.assertFalse(ok)
            current = root / "experiments" / "training_logs" / "current.log"
            baseline = root / "experiments" / "training_logs" / "baseline.log"
            self.assertEqual(current.read_text(), logfile.read_text())
            self.assertEqual(baseline.read_text(), logfile.read_text())

    def test_successful_plot_import_is_called(self):
        calls = []

        def fake_import(name):
            if name == "matplotlib":
                return SimpleNamespace(use=lambda *args, **kwargs: None)
            if name == "experiments.plot_metrics":
                return SimpleNamespace(
                    plot_comparison=lambda *args, **kwargs: calls.append(args) or True
                )
            raise ImportError(name)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            logfile = root / "run.txt"
            logfile.write_text("step:0/1 val_loss:1.0 val_bpb:1.0\n")

            with mock.patch("experiments.plotting_hook.importlib.import_module", side_effect=fake_import):
                ok = maybe_update_experiment_plots(logfile, root=root)

        self.assertTrue(ok)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
