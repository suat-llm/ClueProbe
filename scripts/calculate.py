"""Offline result coverage audit or frozen Y2/Y3 statistics reproduction."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser(description=__doc__,add_help=('--workflow' not in sys.argv))
 p.add_argument('--workflow',choices=['coverage','y2y3'],default='coverage')
 a,rest=p.parse_known_args()
 target='statistics/reproduce_y2y3.py' if a.workflow=='y2y3' else 'tools/audit_results.py'
 raise SystemExit(subprocess.call([sys.executable,str(ROOT/target),*rest]))
if __name__=='__main__':main()
