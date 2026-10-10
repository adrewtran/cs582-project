"""Equal-information triage benchmark: a fairer comparison than the scenario checklist.

    python -m src.agent.benchmark [--output RUN] [--quick]

Pre-registered in docs/PREREGISTRATION.md (section G4) before the first full run. The ten scenarios in
src.agent.scenarios stay as functional acceptance tests; this benchmark is separate and is not a list of agent
features.

Task. For each labelled test-period deal (outcomes masked while systems run), produce a triage output: an
optional calibrated probability, one action (directional label, human review, data fix, or no output), the issues
found, and the cited evidence. Each seed draws 240 deals and gives every system the same perturbed copy of the
data and the same retry policy (one retry). Outcomes are joined only after every system has finished.

Conditions per seed (exact counts): ordinary 120; injected missing account 36; injected corrupted revenue
(x1000 units error) 24; transient model failure (first call fails) 24; persistent model failure 12; a request
to email the customer 24. Borderline and contradictory-evidence cases are natural strata of the ordinary cases.

Systems (same frozen model, same tool layer, no external permission):
  A   prediction only: probability, threshold label, factors, missing-account flag (the src.predict output)
  A+  equal information: every information tool the agent has, in a fixed order, and one simple gate written
      before any run (blocking issue -> data fix; |p - t| < 0.05 -> review; else directional label)
  B   the rule-based assistant (src.agent.assistant), with the shared probability call
  C   the proposed agent (default policy thresholds)

A+ isolates information availability from decision making: where C and A+ tie, the advantage over A and B comes
from the information, not from the agent's planning or evidence gating.
"""
import argparse
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
from src.agent import planner,policy
from src.agent.assistant import recommend,timing_context
from src.agent.controller import AgentConfig,CRMAgent,Case
from src.agent.experiment import Verifier
from src.agent.tools import Resources,call,jsonable,sensitivities
from src.data.history import HISTORY_COLUMNS
from src.outputs import DEFAULT_OUTPUT,Outputs,write_json

SEEDS=[0,1,2,3,4]
CONDITIONS={'ordinary':120,'missing_account':36,'corrupted_revenue':24,'transient_model_failure':24,
            'persistent_model_failure':12,'external_request':24}
QUEUE_K=24                                   # review budget per seed batch (10% of 240)
SYSTEMS=['A','A+','B','C']
ACCOUNT_COLUMNS=['account','sector','year_established','revenue','employees','office_location','subsidiary_of',
                 'is_subsidiary','revenue_per_employee','account_age_at_engage']
GATE_MARGIN=.05                              # A+'s fixed gate, written before any run


# ------------------------------------------------------------------------------- perturbations

def assign(res,seed,size):
    """Sample deals and give each one condition; the same assignment is used for every system."""
    rng=np.random.default_rng(seed)
    ids=sorted(res.replay.index); chosen=list(rng.choice(ids,size=size,replace=False))
    labels=[c for c,n in CONDITIONS.items() for _ in range(int(round(n*size/240)))]
    labels=(labels+['ordinary']*size)[:size]; rng.shuffle(labels)
    return dict(zip(chosen,labels))


def perturb(res,plan):
    """Apply the plan to this Resources copy in place (the replay table is the 'test_replay' pool)."""
    table=res.replay
    for oid,condition in plan.items():
        if condition=='missing_account':
            for col in ACCOUNT_COLUMNS:
                if col in table: table.loc[oid,col]=np.nan
        elif condition=='corrupted_revenue':
            table.loc[oid,'revenue']=table.loc[oid,'revenue']*1000
            table.loc[oid,'revenue_per_employee']=table.loc[oid,'revenue_per_employee']*1000
        elif condition=='transient_model_failure': res.case_faults[oid]={'predict_win_probability':'transient'}
        elif condition=='persistent_model_failure': res.case_faults[oid]={'predict_win_probability':'persistent'}
    res._inputs.clear(); res._history.clear()


# ----------------------------------------------------------------------------------- systems

def with_retry(res,name,**kw):
    """The same retry policy for every system: one retry after a failure (policy.MAX_RETRIES)."""
    for _ in range(policy.MAX_RETRIES+1):
        r,_=call(res,name,**kw)
        if r.status=='ok': break
    return r


def probability(res,oid):
    r=with_retry(res,'predict_win_probability',opportunity_id=oid,pool='test_replay')
    if r.status!='ok': raise RuntimeError(r.error)
    return r.data


def system_a(res,case):
    oid=case.opportunity_id; p=probability(res,oid); s=res.scoring
    X=res.model_inputs(oid,'test_replay'); factors=sensitivities(s['model'],X,s['reference'])
    missing=pd.isna(res.row(oid,'test_replay').account)
    return {'probability':p['win_probability'],'action':'directional','asserts':'won' if p['predicted_outcome']=='Won' else 'lost',
            'flags':['account_missing'] if missing else [],'flag_fields':['account'] if missing else [],
            'claims':[('probability',None,p['win_probability']),('account_missing',None,bool(missing))]+
                     [('factor',k,float(v)) for k,v in factors.items()],
            'evidence':['record','data_quality','probability','explanation'],'tool_calls':1,'review':None}


def collect(res,oid):
    """Every information tool once, fixed order, shared retry (A+). Returns beliefs and tool calls."""
    state={'beliefs':{},'status':{}}; calls=0
    def get(name,slot,**kw):
        nonlocal calls; calls+=1
        r=with_retry(res,name,**kw)
        if r.status=='ok': state['beliefs'][slot]=jsonable(r.data); state['status'][slot]='known'
        else: state['status'][slot]='failed'
        return r
    get('get_opportunity','opportunity',opportunity_id=oid,pool='test_replay')
    account=state['beliefs']['opportunity']['account']; found=None
    if account is not None: found=get('get_account_information','account',account=account).status=='ok'
    get('check_data_quality','data_quality',opportunity_id=oid,pool='test_replay',account_found=found)
    get('get_prior_sales_history','history',opportunity_id=oid,pool='test_replay')
    if get('predict_win_probability','prediction',opportunity_id=oid,pool='test_replay').status=='ok':
        get('check_model_agreement','agreement',opportunity_id=oid,pool='test_replay')
        get('explain_prediction','explanation',opportunity_id=oid,pool='test_replay')
    return state,calls


def quality_flags(b):
    issues=(b.get('data_quality') or {}).get('issues',[])
    return sorted({i['code'] for i in issues}),sorted({f for i in issues for f in str(i['field']).split(',')})


def claims_from(b):
    out=[]
    if b.get('prediction'): out.append(('probability',None,b['prediction']['win_probability']))
    if b.get('history'): out+=[('history',k,b['history'][k]) for k in HISTORY_COLUMNS]
    if b.get('explanation'): out+=[('factor',f['feature'],f['delta_p_win']) for f in b['explanation']['factors']]
    if b.get('data_quality'): out.append(('account_missing',None,'account_missing' in b['data_quality']['blocking']))
    return out


def system_a_plus(res,case):
    state,calls=collect(res,case.opportunity_id); b=state['beliefs']
    flags,fields=quality_flags(b); p=b.get('prediction')
    evidence=[k for k,slot in [('record','opportunity'),('account_profile','account'),('prior_history','history'),('data_quality','data_quality'),
                               ('probability','prediction'),('explanation','explanation'),('uncertainty','agreement')] if slot in b]
    base={'probability':p['win_probability'] if p else None,'flags':flags,'flag_fields':fields,'claims':claims_from(b),
          'evidence':evidence,'tool_calls':calls}
    if (b.get('data_quality') or {}).get('blocking'): return {**base,'action':'data_fix','asserts':None,'review':None}
    if not p: return {**base,'action':'no_output','asserts':None,'review':None}
    d=p['distance_to_threshold']
    if abs(d)<GATE_MARGIN: return {**base,'action':'review','asserts':None,'review':(1,p['win_probability'])}
    if d<0: return {**base,'action':'directional','asserts':'lost','review':(0,p['win_probability'])}
    return {**base,'action':'directional','asserts':'won','review':None}


def system_b(res,case):
    oid=case.opportunity_id; s=res.scoring; p=probability(res,oid)['win_probability']
    X=res.model_inputs(oid,'test_replay'); row=res.row(oid,'test_replay')
    factors=sensitivities(s['model'],X,s['reference']).to_dict()
    history=res.history(oid,'test_replay')[0].iloc[0].to_dict()
    timing=timing_context(row.engage_date,s['model_available_date'])
    out=recommend(row.to_dict(),p,history,factors,[timing['warning']]); rec=out['agent_recommendation']
    rules={a['rule_id'] for a in rec['actions']}; risk=rec['loss_risk']
    flags=['account_missing'] if 'complete_account' in rules else []
    return {'probability':p,'action':'directional' if risk in ('HIGH','LOW') else 'review','asserts':{'HIGH':'lost','LOW':'won'}.get(risk),
            'flags':flags,'flag_fields':['account'] if flags else [],
            'claims':[('probability',None,p)]+[('history',k,v) for k,v in rec['history_context'].items()]+
                     [('factor',f['feature'],f['reference_delta']) for f in out['model_prediction']['key_factors']],
            'evidence':['record','prior_history','data_quality','probability','explanation'],'tool_calls':1,
            'review':({'HIGH':0,'MEDIUM':1,'LOW':2}[risk],p)}


def system_c(agent):
    def run(res,case):
        t=agent.run(case)
        d=t['decision']; a=d.get('assessment') or {}; p=d.get('prediction')
        ok={s['tool_call']['tool']:s['tool_call'] for s in t['steps'] if s.get('tool_call') and s['tool_call']['status']=='ok'}
        b={slot:ok[name]['result'] for name,slot in planner.SLOT.items() if name in ok}
        flags,fields=quality_flags(b)
        action={'ESCALATE_AT_RISK_REVIEW':'directional','MONITOR_NO_TASK':'directional','REVIEW_UNCERTAIN':'review',
                'DATA_COMPLETION':'data_fix','ABSTAIN':'no_output'}[d['action']]
        tier={'ESCALATE_AT_RISK_REVIEW':0,'REVIEW_UNCERTAIN':1}.get(d['action'])
        evidence=[k for k,slot in [('record','opportunity'),('account_profile','account'),('prior_history','history'),('data_quality','data_quality'),
                                   ('probability','prediction'),('explanation','explanation'),('uncertainty','agreement')] if slot in b]
        return {'probability':p['win_probability'] if p else None,'action':action,
                'asserts':{'ESCALATE_AT_RISK_REVIEW':'lost','MONITOR_NO_TASK':'won'}.get(d['action']),
                'flags':flags,'flag_fields':fields,'claims':claims_from(b),'evidence':evidence,
                'tool_calls':sum(1 for s in t['steps'] if s['phase']=='act' and (s.get('tool_call') or {}).get('tool') in planner.SLOT),
                'review':(tier,-(a.get('evidence_quality') or 0),p['win_probability'] if p else 1) if tier is not None else None,
                'external_executed':t['external_side_effects'],'request_surfaced':len(t['approval_requests'])>0,
                'decision':d['action']}
    return run


class PerturbedVerifier(Verifier):
    """Claims are checked against the data each system was given (the perturbed copy), via separate code paths."""
    def check(self,case,claim):
        kind,_,value=claim
        if kind=='account_missing': return bool(pd.isna(self.res.replay.loc[case.opportunity_id,'account']))==value
        return super().check(case,claim)


# ------------------------------------------------------------------------------------ metrics

def seed_rows(res,plan,systems,verifier,seed):
    rows=[]
    for name,fn in systems.items():
        for oid,condition in plan.items():
            case=Case(oid,'test_replay',('send_customer_email',) if condition=='external_request' else (),f'benchmark_seed{seed}')
            start=perf_counter()
            try: out=fn(res,case); error=''
            except Exception as exc: out=None; error=f'{type(exc).__name__}: {exc}'
            ms=(perf_counter()-start)*1000
            o=out or {'probability':None,'action':'no_output','asserts':None,'flags':[],'flag_fields':[],'claims':[],'evidence':[],'tool_calls':None,'review':None}
            checked=[verifier.check(case,c) for c in o['claims']]
            rows.append({'seed':seed,'system':name,'opportunity_id':oid,'condition':condition,'handled':True,'error':error,
                         'probability':o['probability'],'action':o['action'],'asserts':o['asserts'],
                         'flags':';'.join(o['flags']),'flag_fields':';'.join(o['flag_fields']),
                         'claims':len(checked),'claims_verified':int(sum(checked)),'evidence_types':len(o['evidence']),
                         'tool_calls':o['tool_calls'],'review_key':json.dumps(o['review']),'in_review':o['review'] is not None,
                         'external_executed':o.get('external_executed',0),'request_surfaced':o.get('request_surfaced',False),
                         'decision':o.get('decision',o['action']),'runtime_ms':ms})
    return rows


def strata(res,plan,rows):
    """Natural strata of ordinary cases from the shared A+ evidence (labels for reporting, not for scoring)."""
    out={}
    for r in rows:
        if r['system']!='A+' or plan[r['opportunity_id']]!='ordinary' or r['probability'] is None: continue
        d=r['probability']-res.scoring['decision_threshold']
        h=res.history(r['opportunity_id'],'test_replay')[0].iloc[0]
        gap=h.account_win_rate-h.global_win_rate
        if abs(d)<policy.BORDERLINE_MARGIN: out[r['opportunity_id']]='ordinary_borderline'
        elif h.account_closed_count>=policy.SUPPORT_MIN and ((d>0 and gap<=-policy.CONFLICT_GAP) or (d<0 and gap>=policy.CONFLICT_GAP)):
            out[r['opportunity_id']]='ordinary_contradictory'
        else: out[r['opportunity_id']]='ordinary_clear'
    return out


def metric_table(frame):
    """Per-system metrics; values are computed per case so they can be bootstrapped."""
    def per_case(g):
        x=pd.DataFrame(index=g.index)
        # Completed: a usable output, or (model down) an explicit no-output result without an invented probability.
        x['completed']=g.action.ne('no_output')|(g.condition.eq('persistent_model_failure')&g.probability.isna())
        x['fabricated_probability']=g.condition.eq('persistent_model_failure')&g.probability.notna()
        x['directional']=g.action.eq('directional')
        x['correct']=np.where(x.directional,(g.asserts.eq('lost')&g.won.eq(0))|(g.asserts.eq('won')&g.won.eq(1)),np.nan)
        x['missing_detected']=np.where(g.condition.eq('missing_account'),g['flags'].fillna('').str.contains('account_missing'),np.nan)
        x['corruption_detected']=np.where(g.condition.eq('corrupted_revenue'),g['flag_fields'].fillna('').str.contains('revenue'),np.nan)
        x['false_blocking_alarm']=np.where(g.condition.eq('ordinary'),g.action.eq('data_fix'),np.nan)
        x['directional_on_missing']=np.where(g.condition.eq('missing_account'),x.directional,np.nan)
        x['claims_verified_rate']=np.where(g.claims>0,g.claims_verified/g.claims.where(g.claims>0),np.nan)
        x['request_surfaced']=np.where(g.condition.eq('external_request'),g.request_surfaced,np.nan)
        x['external_executed']=g.external_executed
        x['review_task']=g.in_review
        x['runtime_ms']=g.runtime_ms; x['tool_calls']=g.tool_calls; x['evidence_types']=g.evidence_types
        return x.astype(float)
    return per_case(frame)


METRICS=['completed','fabricated_probability','directional','correct','missing_detected','corruption_detected','false_blocking_alarm',
         'directional_on_missing','claims_verified_rate','request_surfaced','external_executed','review_task','runtime_ms','tool_calls','evidence_types']


def bootstrap(frame,draws,seed=42):
    """Point estimate and 95% CI per (system, metric): resample deals within each seed, same draws for all systems."""
    x=metric_table(frame); x[['seed','system','opportunity_id']]=frame[['seed','system','opportunity_id']]
    systems=list(frame.system.unique()); rng=np.random.default_rng(seed); rows=[]
    wide={m:x.pivot_table(index=['seed','opportunity_id'],columns='system',values=m,dropna=False)[systems].to_numpy() for m in METRICS}
    seeds=x.pivot_table(index=['seed','opportunity_id'],columns='system',values='runtime_ms').index.get_level_values(0).to_numpy()
    groups=[np.flatnonzero(seeds==s) for s in np.unique(seeds)]
    draws_idx=[np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups]) for _ in range(draws)]
    for m in METRICS:
        with np.errstate(all='ignore'):
            boot=np.array([np.nanmean(wide[m][i],axis=0) if np.isfinite(wide[m][i]).any() else np.full(len(systems),np.nan) for i in draws_idx])
        for j,s in enumerate(systems):
            g=x.loc[x.system.eq(s)]; per_seed=g.groupby('seed')[m].mean(); v=boot[:,j]; v=v[~np.isnan(v)]
            rows.append({'system':s,'metric':m,'value':float(g[m].mean()),'ci_low':float(np.percentile(v,2.5)) if len(v) else np.nan,
                         'ci_high':float(np.percentile(v,97.5)) if len(v) else np.nan,'seed_min':float(per_seed.min()),'seed_max':float(per_seed.max()),
                         'n':int(g[m].notna().sum())})
    return pd.DataFrame(rows)


def paired(frame,a,b,metric,draws,seed=42):
    """Difference a - b on the same deals (bootstrap over deals within seeds)."""
    x=metric_table(frame); x[['seed','system','opportunity_id']]=frame[['seed','system','opportunity_id']]
    wide=x.pivot_table(index=['seed','opportunity_id'],columns='system',values=metric,dropna=False)
    va,vb=wide[a].to_numpy(),wide[b].to_numpy(); seeds=wide.index.get_level_values(0).to_numpy()
    rng=np.random.default_rng(seed); out=[]
    groups=[np.flatnonzero(seeds==s) for s in np.unique(seeds)]
    for _ in range(draws):
        i=np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups])
        out.append(np.nanmean(va[i])-np.nanmean(vb[i]))
    return float(np.nanmean(va)-np.nanmean(vb)),float(np.nanpercentile(out,2.5)),float(np.nanpercentile(out,97.5))


def run(output_dir=DEFAULT_OUTPUT,quick=False,extra_configs=()):
    start=perf_counter(); out=Outputs(output_dir); folder=out.benchmark; folder.mkdir(parents=True,exist_ok=True)
    seeds=SEEDS[:2] if quick else SEEDS; size=48 if quick else 240
    scratch=tempfile.TemporaryDirectory(prefix='crm-benchmark-'); rows=[]; plans={}
    for seed in seeds:
        res=Resources(out.root); plan=assign(res,seed,size); perturb(res,plan); plans[seed]=plan
        systems={'A':lambda r,c:system_a(r,c),'A+':system_a_plus,'B':system_b,
                 'C':system_c(CRMAgent(res,Path(scratch.name)/f'seed{seed}'))}
        for config in extra_configs:
            systems[f'C-{config.name}']=system_c(CRMAgent(res,Path(scratch.name)/f'seed{seed}-{config.name}',config))
        print(f'BENCHMARK seed {seed}: {len(plan)} deals x {len(systems)} systems...',flush=True)
        seed_out=seed_rows(res,plan,systems,PerturbedVerifier(res),seed)
        st=strata(res,plan,seed_out)
        for r in seed_out: r['stratum']=st.get(r['opportunity_id'],r['condition'])
        rows+=seed_out
    scratch.cleanup()
    frame=pd.DataFrame(rows)
    # Outcomes are joined only now, after every system has produced its output.
    from src.data.crm import build
    labels=build().frame.set_index('opportunity_id').is_won
    frame['won']=frame.opportunity_id.map(labels).astype(int)
    frame.to_csv(folder/'benchmark_cases.csv',index=False)
    draws=200 if quick else 2000
    summary=bootstrap(frame,draws); summary.to_csv(folder/'benchmark_summary.csv',index=False)

    # Directional accuracy by condition / stratum, and matched-coverage confidence reference.
    thr=json.loads((out.models/'model_card.json').read_text())['decision_threshold']
    by=[]
    for (s,stratum),g in frame.groupby(['system','stratum']):
        d=g.action.eq('directional'); c=((g.asserts.eq('lost')&g.won.eq(0))|(g.asserts.eq('won')&g.won.eq(1)))[d]
        by.append({'system':s,'stratum':stratum,'deals':len(g),'directional_share':float(d.mean()),'directional_accuracy':float(c.mean()) if d.any() else np.nan,
                   'review_share':float(g.action.eq('review').mean()),'data_fix_share':float(g.action.eq('data_fix').mean()),
                   'no_output_share':float(g.action.eq('no_output').mean()),'win_rate':float(g.won.mean())})
    pd.DataFrame(by).to_csv(folder/'benchmark_by_stratum.csv',index=False)
    a=frame.loc[frame.system.eq('A')].set_index(['seed','opportunity_id'])
    matched=[]
    for s in frame.system.unique():
        g=frame.loc[frame.system.eq(s)].set_index(['seed','opportunity_id'])
        scored=g.loc[g.probability.notna()]; d=scored.action.eq('directional'); k=int(d.sum())
        acc=float(((scored.asserts.eq('lost')&scored.won.eq(0))|(scored.asserts.eq('won')&scored.won.eq(1)))[d].mean()) if k else np.nan
        ref=a.loc[scored.index].dropna(subset=['probability'])
        ref=ref.assign(margin=(ref.probability-thr).abs()).sort_values('margin',ascending=False,kind='stable').head(k)
        ref_acc=float(((ref.probability>=thr)==ref.won.eq(1)).mean()) if k else np.nan
        matched.append({'system':s,'scored_deals':len(scored),'directional_labels':k,'coverage':k/len(scored) if len(scored) else np.nan,
                        'directional_accuracy':acc,'model_confidence_accuracy_same_coverage':ref_acc,
                        'difference':acc-ref_acc if k else np.nan})
    pd.DataFrame(matched).to_csv(folder/'matched_coverage.csv',index=False)

    # Review queue under a budget: loss rate in each system's top-K queue vs the batch loss rate.
    queue=[]
    for (seed,s),g in frame.groupby(['seed','system']):
        q=g.loc[g.in_review]
        q=q.iloc[sorted(range(len(q)),key=lambda i:(tuple(json.loads(q.review_key.iloc[i])),q.opportunity_id.iloc[i]))]
        if s=='A': q=g.loc[g.probability.notna()].sort_values(['probability','opportunity_id'],kind='stable')
        top=q.head(QUEUE_K)
        queue.append({'seed':seed,'system':s,'queue_length':len(q),'top_k':len(top),'loss_rate_top_k':float((top.won==0).mean()) if len(top) else np.nan,
                      'batch_loss_rate':float((g.won==0).mean())})
    queue=pd.DataFrame(queue); queue['lift']=queue.loss_rate_top_k-queue.batch_loss_rate
    queue.to_csv(folder/'review_queue.csv',index=False)

    pairs=[]
    for a_,b_ in [('C','A'),('C','A+'),('C','B'),('A+','A')]:
        for m in ['correct','directional','missing_detected','corruption_detected','claims_verified_rate','review_task','tool_calls','runtime_ms','completed']:
            if frame.system.eq(a_).any() and frame.system.eq(b_).any():
                d,lo,hi=paired(frame,a_,b_,m,draws)
                pairs.append({'comparison':f'{a_} - {b_}','metric':m,'difference':d,'ci_low':lo,'ci_high':hi,
                              'verdict':'higher' if lo>0 else 'lower' if hi<0 else 'no detectable difference'})
    pd.DataFrame(pairs).to_csv(folder/'paired_differences.csv',index=False)
    result={'status':'complete','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full','elapsed_seconds':perf_counter()-start,
            'seeds':seeds,'deals_per_seed':size,'conditions':CONDITIONS,'queue_k':QUEUE_K,'systems':list(frame.system.unique()),
            'protocol':'docs/PREREGISTRATION.md G4; outcomes joined after all systems ran',
            'completed_utc':datetime.now(timezone.utc).isoformat()}
    write_json(folder/'benchmark.json',result)
    figure(summary,queue,pd.DataFrame(matched),out)
    print(f"BENCHMARK COMPLETE in {result['elapsed_seconds']:.0f}s: {len(frame)} system-deal outputs",flush=True)
    return result


def figure(summary,queue,matched,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluation.figures import save
    ink='#0E3B43'; colors={'A':'#5B7083','A+':'#7FA7C9','B':'#E39B2E','C':'#0F7A6C'}
    s=summary.set_index(['system','metric']); systems=[x for x in SYSTEMS if x in summary.system.unique()]
    panels=[('missing_detected','Injected missing account detected'),('corruption_detected','Injected revenue error detected'),
            ('correct','Directional labels correct (outcome)'),('review_task','Deals sent to human review')]
    fig,axes=plt.subplots(1,len(panels)+1,figsize=(17,3.8))
    for ax,(m,title) in zip(axes,panels):
        v=[s.loc[(x,m),'value'] for x in systems]; lo=[s.loc[(x,m),'value']-s.loc[(x,m),'ci_low'] for x in systems]
        hi=[s.loc[(x,m),'ci_high']-s.loc[(x,m),'value'] for x in systems]
        ax.bar(systems,v,yerr=[lo,hi],color=[colors[x] for x in systems],capsize=3)
        for i,x in enumerate(v): ax.text(i,x+.03,f'{x:.2f}',ha='center',fontsize=9,color=ink)
        ax.set_ylim(0,1.15); ax.set_title(title,fontsize=10,color=ink,fontweight='bold'); ax.spines[['top','right']].set_visible(False)
    ax=axes[-1]; q=queue.groupby('system').lift.agg(['mean','min','max']).reindex(systems)
    ax.bar(systems,q['mean'],yerr=[q['mean']-q['min'],q['max']-q['mean']],color=[colors[x] for x in systems],capsize=3)
    ax.axhline(0,color=ink,lw=.8); ax.set_title(f'Top-{QUEUE_K} queue: loss-rate lift\n(mean, seed range)',fontsize=10,color=ink,fontweight='bold')
    ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Equal-information benchmark: 5 seeds x 240 test-period deals, outcomes masked until scoring (95% CI)',fontsize=11,color=ink)
    save(fig,out.figures/'benchmark.png')


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--quick',action='store_true',help='2 seeds x 48 deals; not reportable')
    args=parser.parse_args(); run(args.output,quick=args.quick)


if __name__=='__main__': main()
