"""Forward to frozen three-stage judge (dry-run by default)."""
import subprocess,sys
from pathlib import Path
if __name__=='__main__':
 raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).resolve().parents[1]/'evaluation/judge.py'),*sys.argv[1:]]))
