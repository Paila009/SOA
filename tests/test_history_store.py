import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from history_store import HistoryStore  # noqa: E402


class HistoryStoreTests(unittest.TestCase):
    def test_full_local_analysis_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            store = HistoryStore(Path(directory) / "runs.jsonl")
            record = store.append("my private question", "qwen", {"quantization": "Q4"},
                                  {"decision": "grounded", "support": 0.8, "risk": 0.2},
                                  [{"url": "https://example.test", "snippet": "Evidence"}],
                                  {"total": 10}, "search", answer="A reviewed answer")
            self.assertEqual(record["question"], "my private question")
            self.assertEqual(record["answer"], "A reviewed answer")
            self.assertEqual(record["analysis"]["support"], .8)
            self.assertTrue(record["run_id"])
            self.assertEqual(store.read()[0]["question_id"], record["question_id"])


if __name__ == "__main__":
    unittest.main()
