"""Join original reference annotations while keeping scoring inputs isolated."""
import collections,hashlib,json
from pathlib import Path
P=Path(__file__).parent;ROOT=P.parent;OUT=ROOT/'video_ooc_judged_20261031';OUT.mkdir(exist_ok=True)
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
original=Path.home()/'Downloads/project-23-at-2026-10-19-11-53-8c0e45fd.json'
tasks=json.loads(original.read_text(encoding='utf-8-sig'));tasks={str(t['id']):t for t in tasks}
reference_rows=read(ROOT/'video_ooc_peer_code_audit_20261029/dataset_full_reference.jsonl')
refs={}
for r in reference_rows:
    sid=r['sample_id']
    if sid in refs:continue
    t=tasks[str(r['task_id'])];a=[a for a in t['annotations'] if not a.get('was_cancelled')]
    assert len(a)==1,(sid,'annotation ambiguity')
    values={x['from_name']:x['value'] for x in a[0]['result']}
    def text(k):return '\n'.join(values.get(k,{}).get('text',[])).strip()
    def choice(k):return (values.get(k,{}).get('choices') or [None])[0]
    question=text('question');assert question==r['question'],(sid,'question mismatch')
    label={'Answerable':'A','Guess':'G','Unanswerable':'U'}[r['question_type']]
    assert choice('label')==r['question_type'],(sid,'label mismatch')
    reference=text('guess_answer') if label=='G' else text('answer')
    if label=='A' and not reference:reference=r['reference_answer']
    reference_status='ok'
    if label!='U' and not reference:reference_status='missing_reference'
    elif label=='G' and not text('guess_reasoning'):reference_status='missing_guess_reasoning'
    refs[sid]={'sample_id':sid,'task_id':r['task_id'],'question':question,'label':label,'reference_answer':reference,
               'guess_reasoning':text('guess_reasoning') if label=='G' else '', 'video_id':r['video_id'],
               'video_source':t['data'].get('source'),'scope':choice('requires_causal_inference'),
               'evidence_axis':choice('evidence_axis'),'annotation_note':text('note'),
               'source_annotation_id':a[0]['id'],'reference_status':reference_status,'historically_disputed':str(r['task_id']) in ['13364','14852','13124','13096','13067']}
mapping={'Qwen3-VL-8B':'Qwen3-VL-8B-Instruct','Qwen3-VL-2B':'Qwen3-VL-2B-Instruct',
         'LLaVA-Video-7B':'LLaVA-Video-7B-Qwen2','InternVideo2.5-8B':'InternVideo2.5-Chat-8B'}
base=ROOT/'v11_work_plan_20261030/offline_ready/staged_results';rows=[];sources=[];counts=collections.Counter()
for f in sorted(base.rglob('*.jsonl')):
    sources.append({'path':str(f),'sha256':digest(f)})
    for r in read(f):
        model=mapping.get(r['model_tag'],r['model_tag']);ref=refs[r['sample_id']]
        assert r['question']==ref['question']
        r={**r,'model_tag':model,'id':'|'.join([r['exp'],model,r['prompt_id'],r['input_mode'],r['sample_id'],r['variant']])}
        assert isinstance(r['final_answer'],str)
        rows.append(r);counts[(r['exp'],model,r['prompt_id'])]+=1
assert len(rows)==len({r['id'] for r in rows})==30972
(OUT/'inputs.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
(OUT/'reference_annotations.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in refs.values()),encoding='utf8')
manifest={'original_annotation_file':str(original),'original_sha256':digest(original),'sources':sources,
          'input_sha256':digest(OUT/'inputs.jsonl'),'reference_sha256':digest(OUT/'reference_annotations.jsonl'),
          'rows':len(rows),'label_counts':dict(collections.Counter(r['label'] for r in refs.values())),
          'counts':[{'exp':e,'model':m,'prompt':p,'n':n} for (e,m,p),n in sorted(counts.items())],
          'historical_version_only':True,'disputed_task_ids':['13364','14852','13124','13096','13067'],
          'reference_issues':[r for r in refs.values() if r['reference_status']!='ok'],
          'note':'Preserve historical main set; disputed IDs retained and marked. No claim this is teacher final frozen master.'}
(OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'rows':len(rows),'reference_labels':manifest['label_counts'],'groups':len(counts),'reference_issues':[(r['sample_id'],r['reference_status']) for r in manifest['reference_issues']]},ensure_ascii=False))
