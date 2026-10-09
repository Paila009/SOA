import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from probe_adapter import ProbeAdapter  # noqa: E402


class ProbeAdapterTests(unittest.TestCase):
    def test_rejects_incomplete_signal_record(self):
        with tempfile.TemporaryDirectory() as directory:
            result = ProbeAdapter(directory).validate_signal_record({"logit_entropy": [1.0]})
            self.assertFalse(result["compatible"])
            self.assertIn("hidden_states", result["missing"])

    def test_artifacts_do_not_imply_live_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "logistic_probe.joblib").touch()
            Path(directory, "signal_processor.joblib").touch()
            status = ProbeAdapter(directory).status()
            self.assertTrue(status["artifacts_ready"])
            self.assertFalse(status["live_probability_available"])


if __name__ == "__main__":
    unittest.main()
