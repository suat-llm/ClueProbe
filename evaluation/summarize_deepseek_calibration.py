import collections,csv,hashlib,json,re,zipfile
from calibrate import kappa
from pathlib import Path
P=Path(__file__).parent;O=P/'calibration_deepseek_official'
def read(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
report=json.loads((O/'calibration_report.json').read_text(encoding='utf8'))
b,c=read(O/'behavior.jsonl'),read(O/'content.jsonl')
assert len(b)==len({r['id'] for r in b})==200
assert len(c)==len({r['id'] for r in c})==95
pairs=read(O/'comparison.jsonl');ledger=read(O/'attempt_cost_ledger.jsonl')
inputs={r['id']:r for r in read(P/'original06_inputs.jsonl')}
groups=collections.defaultdict(lambda:([],[]))
for pair in pairs:
    if pair['judge_content'] is None:continue
    answer=inputs[pair['id']]['raw_answer']
    zh=bool(re.search(r'[\u4e00-\u9fff]',answer));en=bool(re.search('[A-Za-z]',answer))
    lang='mixed' if zh and en else 'zh' if zh else 'en' if en else 'other'
    groups[lang][0].append(int(pair['human']['content']));groups[lang][1].append(pair['judge_content']['content_correct'])
report['content_by_original_answer_language']={lang:{'n':len(a),'kappa':kappa(a,b),'agreement':sum(x==y for x,y in zip(a,b))/len(a) if a else None} for lang,(a,b) in sorted(groups.items())}
baseline=json.loads((P/'original06_baseline.json').read_text(encoding='utf8'))
report['requested_model']='deepseek-flash'
report['behavior_confusion']=[{'human':a,'judge':d,'n':n} for (a,d),n in sorted(collections.Counter((r['human']['behavior'],r['judge_behavior']['behavior']) for r in pairs).items())]
report['behavior_disagreement_ids']=[r['id'] for r in pairs if r['human']['behavior']!=r['judge_behavior']['behavior']]
report['content_disagreement_ids']=[r['id'] for r in pairs if r['judge_content'] is not None and int(r['human']['content'])!=r['judge_content']['content_correct']]
report['content_agreement']=(95-len(report['content_disagreement_ids']))/95
report['api_attempts']=len(ledger)
report['judge_error_n']=sum(r['status']!='ok' for r in b+c)
report['token_usage']={k:sum((r.get('usage') or {}).get(k,0) for r in ledger) for k in ['prompt_tokens','completion_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens']}
report['conservative_cost_estimate_cny']=sum(r['conservative_cny'] for r in ledger)
report['cost_note']='Peak uncached official rates used as conservative estimate, not actual billed amount; cache/off-peak can reduce actual charge.'
report['note']='Actual official API calibration completed. Original exported judgments used as the teacher-described baseline; human identity not independently verified. Single candidate only.'
report['returned_model_ids']=sorted({a.get('returned_model') for r in b+c for a in r['attempts'] if a.get('returned_model')})
report['selection_scope']='User-selected official DeepSeek candidate only. Three-provider comparison not performed. Pass does not establish best candidate.'
report['production_started']=False
(O/'calibration_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
kb,kc=report['behavior_kappa'],report['content_kappa'];passed=report['thresholds_passed']
lines=['# DeepSeek官方裁判校准结果','',f'实际完成200条行为、95条内容校准，API尝试{len(ledger)}次，judge_error {report["judge_error_n"]}条。',
       '',f'行为κ {kb:.4f}，一致{200-len(report["behavior_disagreement_ids"])}/200；内容κ {kc:.4f}，一致{95-len(report["content_disagreement_ids"])}/95。',
       '',f'v11门槛：行为κ≥0.8、内容κ≥0.7。本候选：'+('两项均通过。' if passed else '未全部通过；不得开始正式全量评分。'),
       '', '## 语种分组','', '|语种|内容N|κ|一致率|','|---|---:|---:|---:|']
for language,d in report['content_by_language'].items():
    k='不可定义' if d['kappa'] is None else f'{d["kappa"]:.4f}'
    a='不适用' if d['agreement'] is None else f'{d["agreement"]:.1%}'
    lines.append(f'|{language}|{d["n"]}|{k}|{a}|')
lines+=['','按清除开头Answer:/答案:/回答:后的实际字符判语种。小样本语种组不作强结论，0条或单一类别的κ可能无法定义。',
        '', '## 成本与版本','',f'保守费用估计{report["conservative_cost_estimate_cny"]:.4f}元，以高峰、未命中缓存单价核算；实际账单可能因缓存与空闲时段更低。低于本轮1元控制线。',
        '', '服务：api.deepseek.com官方；请求与返回别名均为deepseek-flash；官方模型列表显示DeepSeek-V4.1-Flash。别名不是不可变的日期快照，保留响应与模型列表，不能保证后续服务端版本不变。',
        '', '现有人工基准按老师v11所述单人判定使用，保留来源尚未独立核实说明。没有重新填写人工标签，也没有把修复后的新回答替换进原200条。',
        '', '本次仅完成用户选定DeepSeek的校准，没有完成三家候选择优比较，也没有计算人—人κ；全量评分尚未启动。',
        '', '逐条人工与裁判对照、模型回答、参考答案和理由见judge_calibration.csv。原始API输出和用量在behavior.jsonl、content.jsonl及attempt_cost_ledger.jsonl。']
lines+=['','## 按原始回答语种统计','', '|原始语种|内容N|κ|一致率|','|---|---:|---:|---:|']
for lang,d in report['content_by_original_answer_language'].items():
    lines.append(f'|{lang}|{d["n"]}|{d["kappa"]:.4f}|{d["agreement"]:.1%}|')
lines+=['','上表未清理Answer:等英文前缀，便于对照手册44中文/37英文/14混合的统计。清理前后语种发生变化不会改变裁判请求、判定或总体κ。']
(O/'校准结果说明.md').write_text('\n'.join(lines),encoding='utf8')
freeze={'candidate_qualified':passed,'requested_model':'deepseek-flash','returned_model_ids':report['returned_model_ids'],
        'provider':'https://api.deepseek.com','thinking':{'type':'disabled'},'temperature':0,'max_tokens':256,
        'judge_prompt_version':'v11-A3-A4-unmodified','prompt_file_sha256':hashlib.sha256((P/'v11_prompts.json').read_bytes()).hexdigest(),
        'source_sha256':baseline['source_sha256'],'calibration_baseline_sha256':hashlib.sha256((P/'original06_baseline.json').read_bytes()).hexdigest(),
        'run_signatures':sorted({r['run_signature'] for r in b+c}),'three_provider_selection_completed':False,
        'production_authorized_by_this_file':False,'model_alias_version_mutable':True}
(O/'qualified_candidate_config.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2),encoding='utf8')
with zipfile.ZipFile(P/'DeepSeek官方裁判_200条校准结果_20261031.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in O.iterdir():
        if f.is_file():z.write(f,f.name)
    for n in ['original06_baseline.json','original06_inputs.jsonl','v11_prompts.json','provider_deepseek_official.json','deepseek_available_models.json','deepseek_connection_test.json','run_deepseek_calibration.py','calibrate.py']:
        z.write(P/n,n)
print(json.dumps({k:report[k] for k in ['behavior_n','content_n','behavior_kappa','content_kappa','behavior_agreement','content_agreement','thresholds_passed','api_attempts','judge_error_n','conservative_cost_estimate_cny','behavior_disagreement_ids','content_disagreement_ids','content_by_language']},ensure_ascii=False))
