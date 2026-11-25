from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from video_ooc.manifest import read_jsonl
from video_ooc.evaluation import audit_results
p=argparse.ArgumentParser(description='Audit result coverage, routing, prompts and generation failures without scoring answers.')
p.add_argument('--manifest',type=Path,required=True);p.add_argument('--predictions',type=Path,nargs='+',required=True);p.add_argument('--model-tag',required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert a.output.resolve() not in {a.manifest.resolve(),*(p.resolve() for p in a.predictions)},'Do not overwrite inputs';rows=[r for p in a.predictions for r in read_jsonl(p)];report=audit_results(read_jsonl(a.manifest),rows,a.model_tag);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['model_tag','expected_requests','saved_rows','passed']},indent=2));sys.exit(0 if report['passed'] else 1)
