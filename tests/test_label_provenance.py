import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from label_provenance import validate_response_labels


class LabelProvenanceTests(unittest.TestCase):
    def test_matching_response(self):
        validate_response_labels({'train':[{'id':'1','label':0,
            'data':{'generated_text':'Paris is in France.'},
            'record':{'response':'Paris is in France.','is_hallucinated':0}}]})

    def test_different_response_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Response-label mismatch'):
            validate_response_labels({'train':[{'id':'1','label':0,
                'data':{'generated_text':'London is in France.'},
                'record':{'response':'Paris is in France.','is_hallucinated':0}}]})
