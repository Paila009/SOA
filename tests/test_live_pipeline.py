import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'dashboard'))
from chat_pipeline import ClaimStream, resolve_followup, evidence_id
from jobs import Job, Cancelled
from claim_verifier import verify_claim


class LivePipelineTests(unittest.TestCase):
    def test_stop_closes_inference_connection(self):
        job=Job()
        closed=[]
        job.attach(lambda:closed.append(True))
        job.cancel()
        self.assertEqual(closed,[True])
        with self.assertRaises(Cancelled): job.check()

    def test_cancel_before_attach_cannot_launch_inference(self):
        job=Job();job.cancel()
        with self.assertRaises(Cancelled): job.attach(lambda:None)

    def test_factual_followup_uses_original_subject(self):
        history=[{'role':'user','content':'Who is Messi?'},{'role':'assistant','content':'He is a footballer.'}]
        self.assertEqual(resolve_followup('Compare him with Ronaldo',history),'Compare Messi with Ronaldo')

    def test_snapshot_changes_if_evidence_changes(self):
        self.assertNotEqual(evidence_id([], 'A'),evidence_id([], 'B'))

    @patch('chat_pipeline.verify_claim')
    def test_guard_stops_at_first_failed_claim(self,verify):
        verify.return_value={'score':.01,'status':'unverified','reason':'Missing evidence'}
        stream=ClaimStream(Job(),[],guard=True)
        self.assertFalse(stream('A fabricated statement. Another'))
        self.assertTrue(stream.stopped)
        self.assertEqual(len(stream.reviewed),1)

    @patch('chat_pipeline.verify_claim')
    def test_citation_is_attached_before_review(self,verify):
        verify.return_value={'score':.99,'status':'supported','reason':'Supported'}
        stream=ClaimStream(Job(),[],guard=True)
        stream('Paris is a city.');stream(' [1]');stream(' Next')
        self.assertIn('[1]',stream.reviewed[0]['sentence'])

    @patch('claim_verifier.NLI.predict')
    def test_real_semantic_result_is_used(self,predict):
        predict.return_value={'entailment':.98,'neutral':.01,'contradiction':.01}
        result=verify_claim('A woman is cycling.',[{'snippet':'A woman rides a bicycle.'}])
        self.assertEqual(result['status'],'supported')
        self.assertIn('nli',result['signals'])

    @patch('claim_verifier.NLI.predict')
    def test_conflicting_sources_are_reported(self,predict):
        predict.side_effect=[{'entailment':.98,'neutral':.01,'contradiction':.01},
                             {'entailment':.01,'neutral':.01,'contradiction':.98}]
        result=verify_claim('The museum is open.',[{'snippet':'The museum is open.'},{'snippet':'The museum is not open.'}])
        self.assertEqual(result['status'],'partial')
        self.assertIn('Sources disagree',result['reason'])


if __name__=='__main__': unittest.main()
