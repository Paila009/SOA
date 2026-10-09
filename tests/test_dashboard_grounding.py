"""Regression checks for the dashboard's attributable text-overlap guard."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from server import analyze_grounding, calibrated_probe_threshold, plan_search_queries, split_sentences  # noqa: E402


class GroundingReviewTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{
            "title": "Kendrick Lamar",
            "url": "https://en.wikipedia.org/wiki/Kendrick_Lamar",
            "snippet": (
                "Kendrick Lamar is an American rapper and songwriter. "
                "He was born in Compton, California."
            ),
        }]

    def review(self, answer):
        return analyze_grounding(
            self.sources[0]["snippet"], answer, 0.5, "Kendrick Lamar",
            sources=self.sources,
        )

    def test_supported_claim_links_to_exact_passage(self):
        result = self.review("Kendrick Lamar is an American rapper.")
        self.assertEqual(result["sentences"][0]["status"], "supported")
        self.assertEqual(result["sentences"][0]["source_index"], 1)
        self.assertIn("rapper and songwriter", result["sentences"][0]["evidence_excerpt"])
        self.assertEqual(result["source_reviews"][0]["strong_matches"], 1)

    def test_invented_claim_cannot_borrow_words_from_other_claim(self):
        result = self.review(
            "Kendrick Lamar is an American rapper. "
            "He played football for the New Orleans Saints from 2013 to 2016."
        )
        self.assertEqual(result["unsupported_count"], 1)
        self.assertGreater(result["risk"], 0.5)
        self.assertEqual(result["sentences"][1]["status"], "unverified")
        self.assertEqual(result["sentences"][1]["evidence_excerpt"], "")

    def test_unseen_date_does_not_get_strong_match(self):
        result = self.review("Kendrick Lamar was born in Compton in 2009.")
        self.assertLessEqual(result["sentences"][0]["support"], 0.25)

    def test_single_digit_and_negation_conflicts(self):
        source = [{"title": "Example", "snippet": "The project has 4 models. Paris is the capital of France."}]
        number = analyze_grounding(source[0]["snippet"], "The project has 5 models.", sources=source)
        negation = analyze_grounding(source[0]["snippet"], "Paris is not the capital of France.", sources=source)
        self.assertLessEqual(number["support"], 0.25)
        self.assertLessEqual(negation["support"], 0.25)
        self.assertEqual(number["sentences"][0]["status"], "contradicted")
        self.assertEqual(negation["sentences"][0]["status"], "contradicted")

    def test_wrong_citation_cannot_borrow_other_source(self):
        sources = self.sources + [{"title": "Paris", "snippet": "Paris is the capital of France."}]
        result = analyze_grounding("", "Kendrick Lamar is an American rapper [2].", sources=sources)
        self.assertEqual(result["sentences"][0]["status"], "unverified")

    def test_standalone_source_marker_stays_with_claim(self):
        self.assertEqual(
            split_sentences("Curie studied radioactivity.\n[1][2]"),
            ["Curie studied radioactivity. [1][2]"],
        )

    def test_no_sources_is_unverified(self):
        result = analyze_grounding("", "A factual claim.", 0.5, "claim", sources=[])
        self.assertFalse(result["evidence_available"])
        self.assertEqual(result["unsupported_count"], 1)

    def test_comparison_searches_each_subject(self):
        self.assertEqual(
            plan_search_queries("Compare Marie Curie and Ada Lovelace in terms of their contributions"),
            ["Marie Curie", "Ada Lovelace"],
        )
        self.assertEqual(
            plan_search_queries("What is the difference between Python and Java?"),
            ["Python", "Java"],
        )
        self.assertEqual(
            plan_search_queries("Compare Marie Curie and Ada Lovelace and explain their main contributions."),
            ["Marie Curie", "Ada Lovelace"],
        )
        self.assertEqual(plan_search_queries("What is the capital of France?"), ["France"])

    def test_validation_calibration_does_not_use_test_rows(self):
        rows = [
            {"split": "val", "label": 1, "probability": 0.8},
            {"split": "val", "label": 0, "probability": 0.2},
            {"split": "test", "label": 1, "probability": 0.01},
        ]
        calibration = calibrated_probe_threshold(rows)
        self.assertEqual(calibration["samples"], 2)
        self.assertEqual(calibration["split"], "validation")

    def test_risk_and_coverage_are_reported_separately(self):
        result = self.review("Kendrick Lamar is an American rapper. An unsupported moon claim exists.")
        self.assertIn("evidence_coverage", result)
        self.assertIn("weakest_claim_risk", result)
        self.assertGreater(result["weakest_claim_risk"], 0.5)

    def test_low_query_word_overlap_does_not_inflate_hallucination_risk(self):
        result = analyze_grounding('Paris is the capital of France.',
            'Paris is the capital of France.', question='Identify the administrative centre.')
        self.assertEqual(result['question_relevance'], 0)
        self.assertLess(result['risk'], .1)


if __name__ == "__main__":
    unittest.main()
