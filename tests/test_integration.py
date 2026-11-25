import ast,hashlib,importlib.util,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class IntegrationTests(unittest.TestCase):
 def test_selected_sources_match_original_receipts(self):
  for row in json.loads((ROOT/'metadata/source_hashes.json').read_text()):
   with self.subTest(path=row['target']):
    self.assertEqual(hashlib.sha256((ROOT/row['target']).read_bytes()).hexdigest(),row['original_sha256'])
 def test_prompt_protocols_identical(self):
  def constants(path):
   for n in ast.parse(path.read_text()).body:
    if isinstance(n,ast.Assign) and any(getattr(t,'id','')=='PROMPTS' for t in n.targets):return ast.literal_eval(n.value)
  base=constants(ROOT/'closed_api/prompts.py')
  for path in ['runners/main_text/run_qwen.py','runners/ladder_induce/run_qwen2b.py','training/evaluate.py']:
   self.assertEqual(base,constants(ROOT/path))
 def test_dispatch_help_reaches_real_runner(self):
  for runner,flag in [('qwen-main-text','--model-revision'),('closed-official','--provider'),('qwen2b','--model-path')]:
   p=subprocess.run([sys.executable,str(ROOT/'scripts/get_answers.py'),'--runner',runner,'--help'],capture_output=True,text=True)
   self.assertEqual(p.returncode,0,p.stderr);self.assertIn(flag,p.stdout)
 def test_judge_is_offline_and_excludes_content_reference_from_behavior(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);source=p/'input.jsonl';out=p/'prompts.jsonl'
   source.write_text(json.dumps(dict(id='synthetic',question='QUESTION',raw_answer='ANSWER {question}',label='A',reference_answer='PRIVATE_REFERENCE'))+'\n')
   cmd=[sys.executable,str(ROOT/'scripts/eval.py'),'--stage','behavior','--input',str(source),'--output',str(out)]
   r=subprocess.run(cmd,capture_output=True,text=True);self.assertEqual(r.returncode,0,r.stderr)
   self.assertIn('no API called',r.stdout);text=out.read_text();self.assertNotIn('PRIVATE_REFERENCE',text);self.assertIn('ANSWER {question}',text)
if __name__=='__main__':unittest.main()
