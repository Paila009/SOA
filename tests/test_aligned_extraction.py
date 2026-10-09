import unittest
from pathlib import Path


class AlignedExtractionSourceTests(unittest.TestCase):
    def test_script_labels_exact_annotated_response(self):
        source=(Path(__file__).resolve().parents[1]/'scripts/extract_aligned_signals.py').read_text(encoding='utf-8')
        self.assertIn("'generated_text': answer", source)
        self.assertIn("'label_target': 'exact annotated response'", source)
        self.assertNotIn("model.generate(", source)


if __name__ == '__main__': unittest.main()
