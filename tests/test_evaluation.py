import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from video_ooc.manifest import read_jsonl
from video_ooc.evaluation import join_references,audit_results
from video_ooc.prompts import PROMPTS

class EvaluationTest(unittest.TestCase):
    def setUp(self):
        root=Path(__file__).resolve().parents[1];self.jobs=read_jsonl(root/'data/examples/inference_jobs.jsonl');self.refs=read_jsonl(root/'data/examples/judge_references.jsonl')
        self.rows=[{**j,'model_tag':'demo-model','model_version':'synthetic-v1','inference_status':'ok','final_answer':'final only','raw_answer':'<think>private reasoning</think>final only','thinking':'private reasoning','prompt_text':PROMPTS[j['prompt_id']].format(question=j['question'])} for j in self.jobs]
    def test_versioned_join_preserves_rows_and_excludes_reasoning(self):
        result=join_references(self.rows,self.refs);self.assertEqual(len(result),len(self.rows));self.assertEqual(result[0]['model_answer'],'final only');self.assertNotIn('thinking',result[0]);self.assertNotIn('raw_answer',result[0]);self.assertNotEqual(result[0]['variant'],result[1]['variant'])
    def test_cannot_fall_back_to_raw_thinking(self):
        del self.rows[0]['final_answer']
        with self.assertRaisesRegex(ValueError,'final_answer'):join_references(self.rows,self.refs)
    def test_duplicate_reference_rejected(self):
        with self.assertRaisesRegex(ValueError,'Duplicate reference'):join_references(self.rows,self.refs+self.refs[:1])
    def test_complete_audit_and_missing_duplicate_failure(self):
        self.assertTrue(audit_results(self.jobs,self.rows,'demo-model')['passed'])
        report=audit_results(self.jobs,self.rows[:-1]+self.rows[:1],'demo-model');self.assertFalse(report['passed']);self.assertEqual(len(report['missing']),1);self.assertEqual(len(report['duplicates']),1)
    def test_failure_kept_as_attempt_not_success(self):
        self.rows[0]['inference_status']='truncated';self.rows[0]['final_answer']=''
        result=audit_results(self.jobs,self.rows,'demo-model');self.assertFalse(result['passed']);self.assertEqual(result['unique_attempted_keys'],len(self.jobs));self.assertEqual(len(result['failed_or_empty']),1)
    def test_changed_prompt_or_model_is_detected(self):
        self.rows[0]['prompt_text']='different';self.rows[1]['model_tag']='another';self.assertFalse(audit_results(self.jobs,self.rows,'demo-model')['passed'])
if __name__=='__main__':unittest.main()
