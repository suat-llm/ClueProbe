from pathlib import Path
import argparse,json,sys,collections
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from video_ooc.manifest import read_jsonl,validate_jobs,smoke_jobs
p=argparse.ArgumentParser(description='Validate the LADDER/INDUCE input-only manifest, without model/API calls.')
p.add_argument('manifest',type=Path);p.add_argument('--media-root',type=Path,help='Optional path existence check; relative paths are resolved here.')
a=p.parse_args();rows=read_jsonl(a.manifest);validate_jobs(rows,a.media_root)
print(json.dumps({'requests':len(rows),'smoke_requests':len(smoke_jobs(rows)),'video_paths_checked':a.media_root is not None,'conditions':dict(collections.Counter('/'.join((r['exp'],r['prompt_id'],r['input_mode'])) for r in rows))},indent=2))
