"""Audit saved generation coverage; scientific scoring needs the documented judge."""
import subprocess,sys
from pathlib import Path
if __name__=='__main__':
 raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).resolve().parents[1]/'tools/audit_results.py'),*sys.argv[1:]]))
