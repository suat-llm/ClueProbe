from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from video_ooc.manifest import read_jsonl,write_jsonl
from video_ooc.evaluation import join_references
p=argparse.ArgumentParser(description='Join final answers to reference versions for a separately calibrated judge.')
p.add_argument('--predictions',type=Path,required=True);p.add_argument('--references',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();assert a.output.resolve() not in {a.predictions.resolve(),a.references.resolve()},'Do not overwrite inputs';rows=join_references(read_jsonl(a.predictions),read_jsonl(a.references));write_jsonl(a.output,rows);print(f'Wrote {len(rows)} judge input rows; thinking is not included')
