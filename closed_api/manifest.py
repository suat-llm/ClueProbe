"""Strict v11 routing contract. Reference fields never enter inference jobs."""
import json
from pathlib import Path, PurePosixPath, PureWindowsPath

INPUT_FIELDS = {'exp','sample_id','item','variant','question','video_file','prompt_id','input_mode'}
KEY_FIELDS = ('exp','sample_id','variant','prompt_id','input_mode')
REFERENCE_KEY = ('exp','sample_id','variant')


def read_jsonl(path):
    records=[]
    with Path(path).open(encoding='utf-8-sig') as stream:
        for number,line in enumerate(stream,1):
            if not line.strip():continue
            try:record=json.loads(line)
            except json.JSONDecodeError as exc:raise ValueError(f'{path}:{number}: malformed JSON') from exc
            if not isinstance(record,dict):raise ValueError(f'{path}:{number}: expected an object')
            records.append(record)
    return records


def write_jsonl(path, records):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as stream:
        for record in records:stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n')


def request_key(row):
    return tuple(row[field] for field in KEY_FIELDS)


def reference_key(row):
    return tuple(row[field] for field in REFERENCE_KEY)


def validate_jobs(rows, media_root=None):
    if not rows:raise ValueError('Empty inference manifest')
    seen=set()
    for number,row in enumerate(rows,1):
        if set(row)!=INPUT_FIELDS:raise ValueError(f'row {number}: inference must contain exactly the eight routing/input fields')
        for field in ['sample_id','question']:
            if not isinstance(row[field],str) or not row[field].strip():raise ValueError(f'row {number}: empty {field}')
        key=request_key(row)
        if key in seen:raise ValueError(f'row {number}: duplicate request key {key}')
        seen.add(key)
        if row['prompt_id'] not in {'P0','P1'}:raise ValueError(f'row {number}: unsupported prompt')
        mode=row['input_mode'];variant=row['variant'];exp=row['exp']
        if mode not in {'video','none','swap','black'}:raise ValueError(f'row {number}: unsupported input mode')
        if exp=='LADDER':
            if variant not in {'k0','k1','k2','k3','k4'} or row['prompt_id']!='P0' or mode!='video':raise ValueError(f'row {number}: invalid LADDER condition')
        elif exp=='INDUCE':
            if variant not in {f'v{i}' for i in range(8)}|{'u0','u1','u2'}:raise ValueError(f'row {number}: unknown induction variant')
            expected='swap' if variant=='v6' else 'black' if variant=='v7' else None
            if (expected and mode!=expected) or (not expected and mode not in {'video','none'}):raise ValueError(f'row {number}: variant/input mode mismatch')
        else:raise ValueError(f'row {number}: this manifest contract supports LADDER/INDUCE only; MAIN/TEXT use the documented separate schema')
        video=row['video_file']
        if mode=='none':
            if video is not None:raise ValueError(f'row {number}: text-only input must have null video_file')
        else:
            if not isinstance(video,str) or not video or '\\' in video:raise ValueError(f'row {number}: expected a relative POSIX video path')
            rel=PurePosixPath(video)
            if rel.is_absolute() or PureWindowsPath(video).is_absolute() or '..' in rel.parts:raise ValueError(f'row {number}: unsafe video path')
            if media_root is not None:
                root=Path(media_root).resolve();target=(root/video).resolve()
                # Dataset directories may be legitimate symlinks to shared storage.
                if not target.is_file():raise ValueError(f'row {number}: video does not exist: {root/video}')
    return len(seen)


def smoke_jobs(jobs):
    """Cover every condition plus a complete ladder from each end of the manifest."""
    selected={}
    for row in jobs:selected.setdefault(tuple(row[k] for k in ('exp','variant','prompt_id','input_mode')),row)
    result=list(selected.values());last={}
    for row in jobs:
        if row['exp']=='LADDER':last[row['variant']]=row
    present={request_key(r) for r in result}
    result.extend(row for row in last.values() if request_key(row) not in present)
    return result
