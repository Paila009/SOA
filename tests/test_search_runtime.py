import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
import search_runtime  # noqa: E402


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        search_runtime.clear_retrieval_cache()

    @staticmethod
    def payload():
        return {"query": {"pages": [
            {"title": "Cricket", "fullurl": "https://example/cricket",
             "extract": "Cricket is a bat-and-ball sport played between two teams."},
            {"title": "Cricket (insect)", "fullurl": "https://example/insect",
             "extract": "Crickets are insects related to bush crickets."},
        ]}}

    @patch("search_runtime.urlopen")
    def test_results_are_ranked_and_attributable(self, mocked):
        mocked.return_value = FakeResponse(self.payload())
        sources, metadata = search_runtime.search_web_detailed("cricket sport", 2)
        self.assertEqual(sources[0]["title"], "Cricket")
        self.assertEqual(sources[0]["provider"], "Wikipedia")
        self.assertEqual(metadata["status"], "ok")
        self.assertFalse(metadata["cached"])

    @patch("search_runtime.urlopen")
    def test_second_request_uses_cache(self, mocked):
        mocked.return_value = FakeResponse(self.payload())
        search_runtime.search_web_detailed("cricket", 2)
        _, metadata = search_runtime.search_web_detailed("cricket", 2)
        self.assertTrue(metadata["cached"])
        self.assertEqual(mocked.call_count, 1)

    @patch("search_runtime.time.sleep")
    @patch("search_runtime.urlopen")
    def test_transient_error_retries_then_recovers(self, mocked, _sleep):
        mocked.side_effect = [URLError("temporary"), FakeResponse(self.payload())]
        sources, metadata = search_runtime.search_web_detailed("cricket", 2)
        self.assertEqual(len(sources), 2)
        self.assertEqual(metadata["attempts"], 2)

    @patch("search_runtime.time.sleep")
    @patch("search_runtime.urlopen")
    def test_failure_returns_diagnostic_instead_of_silent_empty(self, mocked, _sleep):
        mocked.side_effect = URLError("blocked")
        sources, metadata = search_runtime.search_web_detailed("cricket", 2)
        self.assertEqual(sources, [])
        self.assertEqual(metadata["status"], "error")
        self.assertIn("blocked", metadata["error"])


if __name__ == "__main__":
    unittest.main()
