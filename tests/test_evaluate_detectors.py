import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_detectors import binary_metrics  # noqa: E402


class EvaluationTests(unittest.TestCase):
    def test_perfect_predictions_have_perfect_metrics(self):
        result = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
        self.assertEqual(result["f1"], 1.0)
        self.assertEqual(result["auroc"], 1.0)


if __name__ == "__main__":
    unittest.main()
