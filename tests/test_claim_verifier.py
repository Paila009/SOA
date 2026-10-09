import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dashboard"))
from claim_verifier import score_passage, split_claims, verify_claim  # noqa: E402


class ClaimVerifierTests(unittest.TestCase):
    def test_bullets_and_citations_stay_auditable(self):
        self.assertEqual(split_claims("• First claim.\n[1]\n• Second claim."),
                         ["First claim. [1]", "Second claim."])

    def test_supported_paraphrase_uses_one_passage(self):
        source = [{"snippet": "Marie Curie was a Polish and naturalized-French physicist and chemist."}]
        result = verify_claim("Marie Curie was a physicist and chemist.", source)
        self.assertEqual(result["status"], "supported")
        self.assertEqual(result["source_index"], 1)

    def test_entity_absence_is_not_a_proven_contradiction(self):
        result = score_passage("London is the capital of France.", "Paris is the capital of France.")
        self.assertEqual(result["status"], "unverified")

    def test_missing_date_is_unverified(self):
        result = score_passage('The mission launched in 2024.', 'The mission launched successfully.')
        self.assertEqual(result['status'], 'unverified')

    def test_date_swap_is_contradicted(self):
        result = score_passage("The mission launched in 2024.", "The mission launched in 2021.")
        self.assertEqual(result["status"], "contradicted")


if __name__ == "__main__":
    unittest.main()
