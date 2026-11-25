"""Structural audits and reference joins, not an uncalibrated substitute judge."""
from .manifest import request_key,reference_key,validate_jobs
from .prompts import PROMPTS


def join_references(predictions, references):
    reference={}
    for row in references:
        key=reference_key(row)
        if key in reference:raise ValueError(f'Duplicate reference version: {key}')
        reference[key]=row
    seen=set();joined=[]
    for row in predictions:
        key=(row['model_tag'],)+request_key(row)
        if key in seen:raise ValueError(f'Duplicate prediction: {key}')
        seen.add(key);ref=reference.get(reference_key(row))
        if ref is None:raise ValueError(f'Missing reference: {reference_key(row)}')
        if row['question']!=ref['question'] or row['item']!=ref['item']:raise ValueError('Reference question/item mismatch')
        if 'final_answer' not in row:raise ValueError('Explicit final_answer required; never fall back to thinking or raw_answer')
        joined.append({'exp':row['exp'],'sample_id':row['sample_id'],'variant':row['variant'],
                       'model_tag':row['model_tag'],'model_version':row.get('model_version'),
                       'prompt_id':row['prompt_id'],'input_mode':row['input_mode'],
                       'question':row['question'],'model_answer':row['final_answer'],
                       'inference_status':row.get('inference_status'),'reference':ref})
    return joined


def audit_results(jobs, predictions, model_tag):
    validate_jobs(jobs);expected={request_key(row):row for row in jobs}
    seen=set();issues=[];duplicates=[];unexpected=[];failed=[]
    for row in predictions:
        try:key=request_key(row)
        except KeyError as exc:issues.append('Missing routing field: '+str(exc));continue
        if row.get('model_tag')!=model_tag:issues.append(f'Unexpected model tag for {key}')
        if key in seen:duplicates.append(key)
        seen.add(key)
        if key not in expected:unexpected.append(key);continue
        source=expected[key]
        if any(row.get(k)!=v for k,v in source.items()):issues.append(f'Input mismatch for {key}')
        if row.get('prompt_text')!=PROMPTS[row['prompt_id']].format(question=row['question']):issues.append(f'Prompt mismatch for {key}')
        final=row.get('final_answer')
        if row.get('inference_status')!='ok' or not isinstance(final,str) or not final.strip():failed.append(key)
        if row.get('finish_reason') in {'length','MAX_TOKENS'}:issues.append(f'Truncated generation for {key}')
        if not row.get('model_version'):issues.append(f'Missing model version for {key}')
    missing=sorted(set(expected)-seen)
    return {'model_tag':model_tag,'expected_requests':len(expected),'saved_rows':len(predictions),
            'unique_attempted_keys':len(seen),'missing':missing,'duplicates':duplicates,
            'unexpected':unexpected,'failed_or_empty':failed,'issues':issues,
            'passed':not any([missing,duplicates,unexpected,failed,issues])}
