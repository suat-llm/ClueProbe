import hashlib,json,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class StatisticsTest(unittest.TestCase):
 def test_dispatch_documents_required_inputs(self):
  r=subprocess.run([sys.executable,str(ROOT/'scripts/calculate.py'),'--workflow','y2y3','--help'],capture_output=True,text=True)
  self.assertEqual(r.returncode,0,r.stderr)
  for flag in ('--package','--inventory','--output'):self.assertIn(flag,r.stdout)
 def test_wrong_package_sha_is_rejected_before_output(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);(p/'bad.zip').write_bytes(b'synthetic');(p/'inventory.json').write_text(json.dumps({'rawzip':{'sha256':'0'*64}}))
   r=subprocess.run([sys.executable,str(ROOT/'statistics/reproduce_y2y3.py'),'--package',str(p/'bad.zip'),'--inventory',str(p/'inventory.json'),'--output',str(p/'out.json')],capture_output=True,text=True)
   self.assertNotEqual(r.returncode,0);self.assertFalse((p/'out.json').exists())
if __name__=='__main__':unittest.main()
