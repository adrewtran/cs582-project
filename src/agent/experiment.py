"""Controlled comparison: prediction-only (A) vs rule-based assistant (B) vs the proposed agent (C), plus ablations.

    python -m src.agent.experiment [--output RUN] [--quick]

All systems see identical cases and the same frozen model. Measured quantities are software/workflow properties
(evidence coverage, verifiable claims, data-issue detection, unsupported labels, manual steps, safety, trace
completeness, determinism, runtime). They are not sales outcomes. The outcome-masked replay on the labelled
test period is reported separately and does not change any choice made by the agent.

Writes to RUN/agent/: scenarios.json, scenario_results.csv, system_comparison.csv, ablation_summary.csv,
open_decisions.csv, replay_outcomes.csv, experiment.json, traces/ and review_tasks/ for the scenario cases,
and two figures in RUN/figures/.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import shutil
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
from src.agent import policy
from src.agent.assistant import recommend,timing_context
from src.agent.controller import ABLATIONS,AgentConfig,CRMAgent,Case,content_hash
from src.agent.scenarios import REVIEW_BUDGET,evaluate_checks,select
from src.agent.tools import ACCOUNT_FIELDS,INFORMATION_TOOLS,Resources,call,jsonable,sensitivities
from src.data.crm import SECTOR_FIXES,load_raw
from src.data.history import HISTORY_COLUMNS,build_history
from src.explain.reference import explain_rows,reference_sensitivities
from src.outputs import DEFAULT_OUTPUT,Outputs,write_json

CHECKLIST=['record','account_profile','prior_history','data_quality','probability','explanation','uncertainty']
ISSUES=['account_missing','retrospective_snapshot','stale_open_record']
STEP_KEYS={'step','timestamp','phase','state_before','evaluation'}
CALL_KEYS={'tool','permission','arguments','status','result','sources','error'}
FINAL_KEYS={'decision','stopping_reason','model','policy_version','goal','steps','content_sha256','final_recommendation'}


# ------------------------------------------------------------------------------ baselines

def context_row(res,oid,pool):
    """Context + model-input row exactly as the traditional predict stage would score it."""
    row=res.row(oid,pool)
    frame=pd.DataFrame([row]).reset_index(drop=True)
    inputs=res.model_inputs(oid,pool)
    for col in inputs: frame[col]=inputs[col].to_numpy()
    return frame


def lookup(res,oid,pool):
    if pool=='open' and oid not in res.open.index:
        raise ValueError(f'{oid} is not a dated Engaging opportunity')
    if pool=='test_replay' and oid not in res.replay.index: raise ValueError(f'{oid} not in replay pool')


def baseline_a(res,case):
    """A: the traditional pipeline row (src.predict / explain_rows): probability, label, priority, factors, warning."""
    lookup(res,case.opportunity_id,case.pool)
    if 'predict_win_probability' in res.faults: raise RuntimeError(res.faults['predict_win_probability'])
    s=res.scoring
    out=explain_rows(s['model'],context_row(res,case.opportunity_id,case.pool),s['features'],s['reference'],s['decision_threshold'])
    return out.iloc[0].to_dict()


def baseline_b(res,case):
    """B: the PR #4 rule-based assistant over the same score, strictly prior history and factors."""
    lookup(res,case.opportunity_id,case.pool)
    if 'predict_win_probability' in res.faults: raise RuntimeError(res.faults['predict_win_probability'])
    s=res.scoring; X=res.model_inputs(case.opportunity_id,case.pool); row=res.row(case.opportunity_id,case.pool)
    p=float(s['model'].predict_proba(X)[0,1])
    factors=sensitivities(s['model'],X,s['reference']).to_dict()   # same numbers as reference_sensitivities, one model call
    history=res.history(case.opportunity_id,case.pool)[0].iloc[0].to_dict()
    timing=timing_context(row.engage_date,s['model_available_date'])
    result=recommend(row.to_dict(),p,history,factors,[timing['warning']])
    result['opportunity_id']=case.opportunity_id
    return result


def run_system(system,res,case,agent=None):
    start=perf_counter()
    try:
        if system=='A': raw=baseline_a(res,case)
        elif system=='B': raw=baseline_b(res,case)
        else: raw=agent.run(case)
        error=''; handled=True
    except ValueError as exc: raw=None; error=f'ValueError: {exc}'; handled=True     # clean validation error
    except Exception as exc: raw=None; error=f'{type(exc).__name__}: {exc}'; handled=False
    return raw,error,handled,round((perf_counter()-start)*1000,3)


# ---------------------------------------------------------------------- normalization

def truth(res,case):
    """Ground-truth data issues, computed directly from the raw-derived tables (not by any system)."""
    issues=set(); row=None
    table=res.open if case.pool=='open' else res.replay
    if case.opportunity_id in table.index:
        row=table.loc[case.opportunity_id]
        if pd.isna(row.account): issues.add('account_missing')
        if row.engage_date<pd.Timestamp(res.scoring['model_available_date']): issues.add('retrospective_snapshot')
        if case.pool=='open' and (res.snapshot_date-row.engage_date).days>res.stale_days: issues.add('stale_open_record')
    return issues


def normalize(system,res,case,raw,error,handled,ms,scenario=''):
    t=truth(res,case)
    o={'system':system,'scenario':scenario,'case_id':case.opportunity_id,'ok':raw is not None and not (system.startswith('C') and raw['decision']['action']=='ABSTAIN'),
       'handled':handled,'error':error,'runtime_ms':ms,'truth_issues':sorted(t),'truth_account_missing':'account_missing' in t,
       'probability':None,'decision':None,'directional_label':False,'evidence_types':[],'flagged_issues':[],
       'routes_to_record_completion':False,'requests_human_review':False,'surfaces_conflict_or_insufficiency':False,
       'task_created':False,'rationale_recorded':False,'audit_record':False,'external_actions_executed':0,'blocked_requests':0,
       'claims':[],'tool_calls':None,'trace_complete':None,'content_hash':None}
    if raw is None: return o
    if system=='A':
        o.update(probability=float(raw['win_probability']),decision=f"{raw['predicted_outcome']}/{raw['priority']}",directional_label=True,
                 evidence_types=['record','data_quality','probability','explanation'],
                 flagged_issues=['account_missing'] if raw['account_missing'] else [])
        o['claims']=[('probability',None,float(raw['win_probability'])),('account_missing',None,bool(raw['account_missing']))]+\
                    [('factor',k,v) for k,v in json.loads(raw['reference_deltas_json']).items()]
        o['content_hash']=content_hash(jsonable(raw))
    elif system=='B':
        pred=raw['model_prediction']; rec=raw['agent_recommendation']; rules={a['rule_id'] for a in rec['actions']}
        warnings=' '.join(rec['warnings'])
        flagged=(['account_missing'] if 'Missing account' in warnings else [])+(['retrospective_snapshot'] if 'did not exist' in warnings else [])
        o.update(probability=pred['win_probability'],decision=f"loss_risk {rec['loss_risk']}",directional_label=rec['loss_risk'] in ('HIGH','LOW'),
                 evidence_types=['record','prior_history','data_quality','probability','explanation'],flagged_issues=flagged,
                 routes_to_record_completion='complete_account' in rules,requests_human_review='human_review' in rules,
                 surfaces_conflict_or_insufficiency=bool({'thin_account_history','product_review','senior_review'}&rules),
                 rationale_recorded=True)
        o['claims']=[('probability',None,pred['win_probability'])]+[('history',k,v) for k,v in rec['history_context'].items()]+\
                    [('factor',f['feature'],f['reference_delta']) for f in pred['key_factors']]
        o['content_hash']=content_hash(jsonable(raw))
    else:
        d=raw['decision']; b={s['tool_call']['tool']:s['tool_call'] for s in raw['steps'] if s.get('tool_call') and s['tool_call']['status']=='ok'}
        a=d.get('assessment') or {}; p=d.get('prediction')
        types=[]
        if 'get_opportunity' in b: types.append('record')
        if 'get_account_information' in b: types.append('account_profile')
        if 'get_prior_sales_history' in b: types.append('prior_history')
        if 'check_data_quality' in b: types.append('data_quality')
        if p: types.append('probability')
        if 'explain_prediction' in b: types.append('explanation')
        if 'check_model_agreement' in b or a.get('model_reliability'): types.append('uncertainty')
        issues=[i['code'] for i in (b.get('check_data_quality',{}).get('result') or {}).get('issues',[])]
        o['error']='; '.join(e['error'] for e in raw['errors'])
        o.update(probability=p['win_probability'] if p else None,decision=d['action'],
                 directional_label=d['action'] in ('ESCALATE_AT_RISK_REVIEW','MONITOR_NO_TASK'),evidence_types=types,
                 flagged_issues=sorted(set(issues)),routes_to_record_completion=d['action']=='DATA_COMPLETION',
                 requests_human_review=d['action'] in ('ESCALATE_AT_RISK_REVIEW','REVIEW_UNCERTAIN'),
                 surfaces_conflict_or_insufficiency=bool(a.get('conflicts')) or (bool(a) and not a.get('sufficient')),
                 task_created=any(x['tool']=='create_review_task' and x['status']=='ok' for x in raw['actions_executed']),
                 rationale_recorded=True,audit_record=True,external_actions_executed=raw['external_side_effects'],
                 blocked_requests=len(raw['blocked_actions']),tool_calls=sum(1 for s in raw['steps'] if s.get('tool_call') and s['tool_call']['status'] in ('ok','error')),
                 trace_complete=trace_complete(raw),content_hash=raw['content_sha256'])
        if p: o['claims'].append(('probability',None,p['win_probability']))
        if 'get_prior_sales_history' in b: o['claims']+=[('history',k,b['get_prior_sales_history']['result'][k]) for k in HISTORY_COLUMNS]
        if 'get_account_information' in b: o['claims']+=[('account',k,b['get_account_information']['result'][k]) for k in ACCOUNT_FIELDS]
        if 'explain_prediction' in b: o['claims']+=[('factor',f['feature'],f['delta_p_win']) for f in b['explain_prediction']['result']['factors']]
        if 'check_data_quality' in b: o['claims'].append(('account_missing',None,'account_missing' in issues))
    o['manual_steps']=len(CHECKLIST)+2-len(o['evidence_types'])-int(o['task_created'])-int(o['rationale_recorded'])
    return o


def trace_complete(trace):
    if not FINAL_KEYS<=set(trace) or not trace['model'].get('bundle_sha256'): return False
    for step in trace['steps']:
        if not STEP_KEYS<=set(step): return False
        if step.get('tool_call') and not CALL_KEYS<=set(step['tool_call']): return False
    return True


# ------------------------------------------------------------------------ independent verifier

class Verifier:
    """Re-derives every cited claim from raw CSVs and the saved bundle, through separate code paths."""

    def __init__(self,res):
        self.res=res; self.accounts=load_raw(res.out.source_data())['accounts'].set_index('account')
        self.pipeline=load_raw(res.out.source_data())['sales_pipeline'].set_index('opportunity_id')
        self.cache={}

    def derived(self,case,kind):
        """Recompute once per case: probability, history and unbatched reference sensitivities."""
        key=(case.opportunity_id,case.pool,kind)
        if key not in self.cache:
            res=self.res; s=res.scoring; oid=case.opportunity_id
            frame=(res.open if case.pool=='open' else res.replay).loc[[oid]].reset_index(drop=True)
            history=build_history(frame[['opportunity_id','engage_date','sales_agent','account','product']],s['history_archive'])[0]
            X=(frame.join(history) if s['feature_set']=='history' else frame)[s['features']]
            self.cache[(oid,case.pool,'history')]=history.iloc[0]
            self.cache[(oid,case.pool,'probability')]=float(s['model'].predict_proba(X)[0,1])
            self.cache[(oid,case.pool,'factor')]=reference_sensitivities(s['model'],X,s['reference']).iloc[0]
        return self.cache[key]

    def check(self,case,claim):
        kind,key,value=claim; oid=case.opportunity_id
        if kind=='probability': return math.isclose(self.derived(case,'probability'),value,abs_tol=1e-9)
        if kind=='account_missing': return pd.isna(self.pipeline.loc[oid,'account'])==value
        if kind=='account':
            raw=self.accounts.loc[self.pipeline.loc[oid,'account'],key]
            if key=='sector': raw=SECTOR_FIXES.get(raw,raw)   # documented spelling fix, disclosed by the tool
            return (pd.isna(raw) and value is None) or str(raw)==str(value) or (not pd.isna(raw) and isinstance(value,(int,float)) and math.isclose(float(raw),float(value)))
        if kind=='history': return math.isclose(float(self.derived(case,'history')[key]),float(value),abs_tol=1e-9)
        if kind=='factor': return math.isclose(float(self.derived(case,'factor')[key]),float(value),abs_tol=1e-9)
        raise ValueError(kind)

    def rate(self,case,claims):
        return [self.check(case,c) for c in claims]


# ------------------------------------------------------------------------ aggregation

def summarize(rows):
    frame=pd.DataFrame(rows); out=[]
    for system,g in frame.groupby('system',sort=False):
        ok=g.loc[g.ok]; claims=g.claims_checked.sum(); verified=g.claims_verified.sum()
        truth=g.truth_issues.map(len).sum(); found=sum(len(set(t)&set(f)) for t,f in zip(g.truth_issues,g.flagged_issues))
        missing=g.loc[g.truth_account_missing]
        borderline=g.loc[g.borderline]
        out.append({'system':system,'cases':len(g),'outputs':len(ok),
            'evidence_types_mean':float(g.evidence_types.map(len).mean()),
            'decisions_with_verified_evidence':float((ok.claims_checked.gt(0)&ok.claims_checked.eq(ok.claims_verified)).mean()) if len(ok) else np.nan,
            'claims_checked':int(claims),'claim_verification_rate':float(verified/claims) if claims else np.nan,
            'data_issue_recall':float(found/truth) if truth else np.nan,
            'missing_account_recall':float(missing.flagged_issues.map(lambda f:'account_missing' in f).mean()) if len(missing) else np.nan,
            'unsupported_labels_out_of_support':int(missing.directional_label.sum()),
            'unsupported_labels_borderline':int(borderline.directional_label.sum()),
            'unsupported_label_rate':float((missing.directional_label.sum()+borderline.directional_label.sum())/max(1,len(missing)+len(borderline))),
            'manual_steps_mean':float(g.manual_steps.mean()),'review_tasks_created':int(g.task_created.sum()),
            'external_actions_executed':int(g.external_actions_executed.sum()),
            'tool_calls_mean':float(g.tool_calls.mean()) if g.tool_calls.notna().any() else np.nan,
            'trace_complete_rate':float(g.trace_complete.astype(float).mean()) if g.trace_complete.notna().any() else 0.,
            'runtime_ms_mean':float(g.runtime_ms.mean()),'runtime_ms_p95':float(g.runtime_ms.quantile(.95))})
    return pd.DataFrame(out)


def evaluate_cases(res,cases,systems,verifier,label,progress=True):
    rows=[]; traces={}
    for name,agent in systems.items():
        if progress: print(f'  {label}: {name} on {len(cases)} cases...',flush=True)
        for case in cases:
            raw,error,handled,ms=run_system('C' if name.startswith('C') else name,res,case,agent)
            o=normalize(name,res,case,raw,error,handled,ms,label)
            o['claims_checked']=len(o['claims']); o['claims_verified']=int(sum(verifier.rate(case,o['claims'])))
            o['borderline']=o['probability'] is not None and abs(o['probability']-res.scoring['decision_threshold'])<policy.BORDERLINE_MARGIN and not o['truth_account_missing']
            rows.append(o)
            if name.startswith('C'): traces[(name,case.opportunity_id)]=raw
    return rows,traces


def replay_check(res_fresh,trace):
    """Re-execute every successful information tool call with its recorded arguments; results must match."""
    matched=0; total=0
    for step in trace['steps']:
        call_=step.get('tool_call')
        if not call_ or call_['status']!='ok' or call_['tool'] not in INFORMATION_TOOLS: continue
        result,_=call(res_fresh,call_['tool'],**call_['arguments'])
        total+=1; matched+=int(jsonable(result.data)==call_['result'])
    return matched,total


def draw(plt,save,s,names,colors,metrics,layout,size,path):
    """Bar panels of the A/B/C comparison: 1x4 for slides, 2x2 for a single paper column."""
    fig,axes=plt.subplots(*layout,figsize=size); axes=axes.ravel()
    for ax,(col,title,sub,top) in zip(axes,metrics):
        ax.bar(names,s[col],color=colors,width=.62)
        for i,v in enumerate(s[col]): ax.text(i,v+top*.02,f'{v:.2f}',ha='center',va='bottom',fontsize=10,fontweight='bold',color='#0E3B43')
        ax.set_ylim(0,top*1.18); ax.set_title(title,fontsize=11,fontweight='bold',color='#0E3B43',pad=16)
        ax.text(.5,1.02,sub,transform=ax.transAxes,ha='center',fontsize=8,color='#5B7083')
        ax.tick_params(labelsize=8.5); ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f"Same {int(s.cases.iloc[0]):,} Engaging deals, same frozen model: workflow properties, not sales outcomes",fontsize=11 if layout[1]==4 else 9.5,color='#0E3B43')
    save(fig,path)


def figure(summary,ablation,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluation.figures import save
    names=['A: prediction\nonly','B: rule\nassistant','C: agent']; colors=['#5B7083','#E39B2E','#0F7A6C']
    s=summary.set_index('system').loc[['A','B','C']]
    metrics=[('unsupported_label_rate','Unsupported directional labels','share of out-of-support / borderline deals; lower is better',1),
             ('data_issue_recall','Known data issues detected','recall; higher is better',1),
             ('manual_steps_mean','Manual steps left (of 9)','lower is better',9),
             ('evidence_types_mean','Evidence types cited (of 7)','agent skips blocked deals on purpose',7)]
    for layout,size,path in [((1,4),(14,3.9),'agent_comparison.png'),((2,2),(7.5,6.6),'agent_comparison_2x2.png')]:
        draw(plt,save,s,names,colors,metrics,layout,size,out.figures/path)
    a=ablation.set_index('configuration').iloc[::-1]
    labels=[n.replace('no_','without ').replace('_',' ') if n!='full_agent' else 'full agent' for n in a.index]
    bar_colors=['#0F7A6C' if n=='full_agent' else '#9CC5BB' for n in a.index]
    fig,axes=plt.subplots(1,3,figsize=(14,3.6))
    for ax,(col,title,fmt) in zip(axes,[('unsupported_label_rate','Unsupported directional labels','{:.0%}'),
                                       ('decisions_changed_vs_full','Decisions changed vs full agent','{:.0%}'),
                                       ('tool_calls_mean','Tool calls per deal','{:.1f}')]):
        values=a[col].fillna(0); ax.barh(labels,values,color=bar_colors)
        top=max(values.max(),1e-9); ax.set_xlim(0,top*1.25)
        for i,v in enumerate(values): ax.text(v+top*.02,i,fmt.format(v),va='center',fontsize=9,color='#0E3B43')
        ax.set_title(title,fontsize=11,fontweight='bold',color='#0E3B43'); ax.tick_params(labelsize=9); ax.spines[['top','right']].set_visible(False)
    for ax in axes[1:]: ax.set_yticklabels([])
    fig.suptitle('Ablations: switch off one component at a time (all Engaging deals)',fontsize=11,color='#0E3B43')
    save(fig,out.figures/'agent_ablation.png')


# ------------------------------------------------------------------------------- main

def run(output_dir=DEFAULT_OUTPUT,quick=False,limit=None):
    start=perf_counter(); out=Outputs(output_dir); folder=out.root/'agent'
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)
    res=Resources(out.root); verifier=Verifier(res)
    scores=pd.read_csv(out.predictions/'open_deal_predictions.csv')
    scenarios=select(res,scores)
    write_json(folder/'scenarios.json',[{k:v for k,v in s.__dict__.items()} for s in scenarios])
    print('AGENT EXPERIMENT: scenarios',[(s.id,s.opportunity_ids[:3]) for s in scenarios],flush=True)

    # 1. Scenarios: identical cases for A, B and C (C writes its traces and tasks under agent/).
    scenario_rows=[]; scenario_traces={}
    for sc in scenarios:
        sres=Resources(out.root,faults=sc.faults) if sc.faults else res
        agent=CRMAgent(sres,folder/'scenario_runs'/sc.id)
        cases=[Case(i,sc.pool,sc.requested_actions,sc.id) for i in sc.opportunity_ids]
        for system in ['A','B','C']:
            if sc.id=='S5' and system=='C':
                t0=perf_counter(); summary,traces=agent.run_batch(cases,review_budget=REVIEW_BUDGET)
                ms=(perf_counter()-t0)*1000/len(cases)
                outputs=[normalize('C',sres,c,next(t for t in traces if t['case']['opportunity_id']==c.opportunity_id),'',True,ms,sc.id) for c in cases]
                queue=summary['sales_review_queue']; scenario_traces[sc.id]=summary
            else:
                outputs=[]
                for c in cases:
                    raw,error,handled,ms=run_system(system,sres,c,agent)
                    outputs.append(normalize(system,sres,c,raw,error,handled,ms,sc.id))
                    if system=='C' and raw: scenario_traces[sc.id]=raw
                queue=None
            checks=evaluate_checks(sc,outputs,queue)
            for o,case in zip(outputs,cases):
                o['claims_checked']=len(o['claims']); o['claims_verified']=int(sum(verifier.rate(case,o['claims'])))
            scenario_rows.append({'scenario':sc.id,'title':sc.title,'kind':sc.kind,'system':system,'cases':len(cases),
                                  'passed':all(checks.values()),**{f'check_{k}':v for k,v in checks.items()},
                                  'decision':outputs[0]['decision'] if len(outputs)==1 else f'{len(outputs)} deals',
                                  'flagged_issues':';'.join(outputs[0]['flagged_issues']) if len(outputs)==1 else '',
                                  'error':outputs[0]['error'] if len(outputs)==1 else '',
                                  'evidence_types':float(np.mean([len(o['evidence_types']) for o in outputs])),
                                  'claims_verified':f"{sum(o['claims_verified'] for o in outputs)}/{sum(o['claims_checked'] for o in outputs)}",
                                  'runtime_ms_mean':float(np.mean([o['runtime_ms'] for o in outputs]))})
    pd.DataFrame(scenario_rows).to_csv(folder/'scenario_results.csv',index=False)
    write_json(folder/'scenario_batch_summary.json',scenario_traces.get('S5',{}))

    # 2. Every Engaging deal: A, B, C and the four ablations on identical inputs.
    ids=list(res.open.sort_values(['engage_date','opportunity_id'],kind='stable').index)
    if quick or limit: ids=ids[:(limit or 40)]
    cases=[Case(i,'open',(),'open_pipeline') for i in ids]
    # Bulk runs really write their tasks and traces, but into a scratch folder: 1,589 deals x 4 configurations
    # would add ~10,000 files. Only the summaries and the scenario traces are kept as evidence.
    scratch=tempfile.TemporaryDirectory(prefix='crm-agent-'); bulk=Path(scratch.name)
    systems={'A':None,'B':None}
    for config in ABLATIONS:
        systems['C' if config.name=='full_agent' else f'C-{config.name}']=CRMAgent(res,bulk/'pipeline_runs'/config.name,config)
    rows,traces=evaluate_cases(res,cases,systems,verifier,'open pipeline')
    summary=summarize(rows)
    names={'C':'full_agent',**{f'C-{c.name}':c.name for c in ABLATIONS}}
    summary.insert(1,'configuration',summary.system.map(lambda s:names.get(s,'baseline')))
    summary.to_csv(folder/'system_comparison.csv',index=False)
    ablation=summary.loc[summary.system.str.startswith('C')].copy()
    full=ablation.loc[ablation.system.eq('C')].iloc[0]
    for col in ['evidence_types_mean','data_issue_recall','unsupported_label_rate','manual_steps_mean','tool_calls_mean','runtime_ms_mean','review_tasks_created']:
        ablation[f'delta_{col}']=ablation[col]-full[col]
    decisions=pd.DataFrame(rows)
    changed={}
    for name in ablation.system:
        a=decisions.loc[decisions.system.eq(name)].set_index('case_id').decision
        b=decisions.loc[decisions.system.eq('C')].set_index('case_id').decision
        changed[name]=float((a!=b.loc[a.index]).mean())
    ablation['decisions_changed_vs_full']=ablation.system.map(changed)
    ablation.to_csv(folder/'ablation_summary.csv',index=False)
    table=decisions.loc[decisions.system.isin(['A','B','C']),['system','case_id','decision','probability','flagged_issues','evidence_types','manual_steps','task_created','runtime_ms']].copy()
    table['flagged_issues']=table.flagged_issues.map(';'.join); table['evidence_types']=table.evidence_types.map(';'.join)
    table.pivot(index='case_id',columns='system',values='decision').add_prefix('decision_').join(
        table.loc[table.system.eq('C')].set_index('case_id')[['probability','flagged_issues','evidence_types']]).reset_index().to_csv(folder/'open_decisions.csv',index=False)

    # 3. Determinism and tool replay on fresh resources (no caches shared with the first run).
    fresh=Resources(out.root); sample=cases[:(20 if quick else 200)]
    again=CRMAgent(fresh,bulk/'determinism_check')
    same=[again.run(c)['content_sha256']==traces[('C',c.opportunity_id)]['content_sha256'] for c in sample]
    replays=[replay_check(fresh,traces[('C',c.opportunity_id)]) for c in sample]
    b_same=[content_hash(jsonable(baseline_b(fresh,c)))==next(r['content_hash'] for r in rows if r['system']=='B' and r['case_id']==c.opportunity_id) for c in sample]

    # 4. Outcome-masked replay on the labelled test period: decisions first, labels joined afterwards.
    replay_ids=list(res.replay.sort_values(['engage_date','opportunity_id'],kind='stable').index)
    if quick or limit: replay_ids=replay_ids[:(limit or 60)]
    agent=CRMAgent(res,bulk/'replay_runs',AgentConfig('full_agent'))
    labels=res.closed.loc[replay_ids,'is_won']; replay_rows=[]
    for oid in replay_ids:
        case=Case(oid,'test_replay')
        a=baseline_a(res,case); b=baseline_b(res,case); c=agent.run(case,write=False)
        replay_rows.append({'opportunity_id':oid,'A_predicted':a['predicted_outcome'],'A_priority':a['priority'],
                            'B_loss_risk':b['agent_recommendation']['loss_risk'],'C_decision':c['decision']['action'],
                            'win_probability':a['win_probability'],'won':int(labels.loc[oid])})
    replay=pd.DataFrame(replay_rows); base=replay.won.mean(); groups=[]
    for col in ['A_predicted','A_priority','B_loss_risk','C_decision']:
        for value,g in replay.groupby(col):
            groups.append({'system_label':col,'group':value,'deals':len(g),'observed_win_rate':float(g.won.mean()),
                           'lift_vs_base_win_rate':float(g.won.mean()-base)})
    pd.DataFrame(groups).to_csv(folder/'replay_outcomes.csv',index=False)
    replay.to_csv(folder/'replay_decisions.csv',index=False)

    scratch.cleanup()
    figure(summary.loc[summary.system.isin(['A','B','C'])],ablation,out)
    manifest=out.read_manifest()
    result={'status':'complete','mode':'SMOKE_TEST_NOT_FINAL' if quick or limit else 'full','policy_version':policy.POLICY_VERSION,
            'completed_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':perf_counter()-start,
            'model':{'feature_set':res.scoring['feature_set'],'name':res.scoring['model'].name,'bundle_sha256':res.bundle_sha256},
            'open_cases':len(cases),'replay_cases':len(replay),'replay_base_win_rate':float(base),
            'scenarios_passed':{s:{sys:bool(r['passed']) for sys,r in g.set_index('system').iterrows()} for s,g in pd.DataFrame(scenario_rows).groupby('scenario')},
            'determinism':{'agent_cases':len(sample),'agent_identical_hashes':int(sum(same)),'rule_assistant_identical':int(sum(b_same))},
            'tool_replay':{'calls':int(sum(t for _,t in replays)),'matched':int(sum(m for m,_ in replays))},
            'claims':'every cited claim re-derived from raw CSVs / saved bundle via separate code paths',
            'interpretation':'Workflow and safety properties under a stated evidence policy. Several checks test capabilities the agent was designed to add; they do not measure sales conversion, revenue or causal uplift.',
            'base_commit':manifest.get('base_commit')}
    write_json(folder/'experiment.json',result)
    manifest.setdefault('stages',{})['agent']=datetime.now(timezone.utc).isoformat()
    manifest['agent_experiment']='agent/experiment.json'
    out.write_manifest(manifest)
    print(f"AGENT EXPERIMENT COMPLETE: {len(cases)} open deals, {len(replay)} replay deals, "
          f"determinism {sum(same)}/{len(sample)}, tool replay {result['tool_replay']['matched']}/{result['tool_replay']['calls']}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT,help='Run folder written by src.run_project')
    parser.add_argument('--quick',action='store_true',help='Small subsets; not reportable')
    args=parser.parse_args()
    run(args.output,quick=args.quick)


if __name__=='__main__': main()
