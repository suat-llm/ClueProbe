"""Dispatch verified inference entry points without importing model runtimes."""
import argparse,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
ENTRIES={
 'intern-supplemental':'runners/supplemental/run_intern.py',
 'qwen-main-text':'runners/main_text/run_qwen.py',
 'qwen2b':'runners/ladder_induce/run_qwen2b.py',
 'qwen8b-native':'runners/ladder_induce/run_native_qwen.py',
 'videollama3':'runners/ladder_induce/run_videollama3.py',
 'closed-official':'closed_api/run_official.py',
 'trained-evaluation':'training/evaluate.py',
}
def main():
 p=argparse.ArgumentParser(description=__doc__,add_help=('--runner' not in sys.argv));p.add_argument('--runner',required=True,choices=ENTRIES)
 a,rest=p.parse_known_args();raise SystemExit(subprocess.call([sys.executable,str(ROOT/ENTRIES[a.runner]),*rest]))
if __name__=='__main__':main()
