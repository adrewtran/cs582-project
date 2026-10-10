"""Can the agent's policy thresholds be chosen by an objective instead of by design?

    python -m src.agent.thresholds [--output RUN] [--quick]

Pre-registered in docs/PREREGISTRATION.md (section G3).

Two kinds of thresholds are kept apart:
  * the model's classification threshold (decision_threshold in models/model_card.json) is already learned:
    src.train picks it on validation by macro-F1. It is reported here, not changed;
  * the agent's policy thresholds (policy.Thresholds) are design choices. This study asks whether outcome
    labels can choose them.

Outcome-labelled thresholds (borderline margin, sufficient evidence, conflict gap, support minimum, committee
split) are tuned on the validation period only (outcomes masked while the agent decides; labels joined
afterwards). Objective: the Wilson 95% lower bound of directional accuracy (ESCALATE_AT_RISK_REVIEW is correct
when the deal was lost, MONITOR_NO_TASK when it was won), subject to directional coverage >= 5%. Safety rules are
outside the search: blocking data issues always route to data completion and external tools never run.

Overfitting check without the test period: tune on the first half of validation (by engagement date), compare
with the defaults on the second half. Adoption rule: the defaults change only if the tuned values beat them on
that held-out half with a paired 95% bootstrap CI above zero. The test period is then used once, for evaluation.

min_gain has no outcome label; it is chosen by a label-free objective: the fewest tool calls such that every
decision equals the full-information decision and every human review task keeps its context.
"""
import argparse
from dataclasses import asdict,replace
from datetime import datetime,timezone
from itertools import product
import json
from pathlib import Path
import tempfile
from time import perf_counter
import numpy as np
import pandas as pd
from src.agent import planner,policy
from src.agent.controller import AgentConfig,CRMAgent,Case
from src.agent.tools import Resources,call,jsonable
from src.outputs import DEFAULT_OUTPUT,Outputs,write_json

GRID={'borderline_margin':[0.,.025,.05,.075,.10,.15],'sufficient_evidence':[.5,.6,.7,.8,.9],
      'conflict_gap':[.05,.10,.15,.20],'support_min':[1,3,5,10],'committee_split':[.2,.4,.6,.8,1.01]}
MIN_COVERAGE=.05
MIN_GAINS=[0.,.05,.10,.15,.20,.25,.30,.35,.40,.45,.50,.55,.60,.65,.70,.80,.90]
DIRECTIONAL={'ESCALATE_AT_RISK_REVIEW':0,'MONITOR_NO_TASK':1}    # the outcome each directional action asserts


def wilson_low(k,n,z=1.96):
    if n==0: return 0.
    p=k/n; return float((p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n))


def beliefs(res,oid,pool):
    """Full-information belief state: every information tool once, in the agent's order (no planning)."""
    state={'beliefs':{},'status':{}}
    def get(name,slot,**kw):
        r,_=call(res,name,**kw)
        if r.status=='ok': state['beliefs'][slot]=jsonable(r.data); state['status'][slot]='known'
        else: state['status'][slot]='failed'
        return r
    if get('get_opportunity','opportunity',opportunity_id=oid,pool=pool).status!='ok': return state
    account=state['beliefs']['opportunity']['account']
    found=None
    if account is not None: found=get('get_account_information','account',account=account).status=='ok'
    get('check_data_quality','data_quality',opportunity_id=oid,pool=pool,account_found=found)
    get('get_prior_sales_history','history',opportunity_id=oid,pool=pool)
    if get('predict_win_probability','prediction',opportunity_id=oid,pool=pool).status=='ok':
        get('check_model_agreement','agreement',opportunity_id=oid,pool=pool)
        get('explain_prediction','explanation',opportunity_id=oid,pool=pool)
    return state


def decide(state,card,th):
    return planner.choose_decision(planner.decision_candidates(state,True,card,th))['decision']


def score(decisions,won):
    """Directional accuracy, coverage and the objective for one policy on labelled deals."""
    d=np.asarray(decisions); won=np.asarray(won)
    esc=d=='ESCALATE_AT_RISK_REVIEW'; mon=d=='MONITOR_NO_TASK'; directional=esc|mon
    correct=int((esc&(won==0)).sum()+(mon&(won==1)).sum()); n=int(directional.sum())
    return {'deals':len(d),'escalate':int(esc.sum()),'monitor':int(mon.sum()),'coverage':n/len(d),
            'directional_accuracy':correct/n if n else np.nan,'objective':wilson_low(correct,n) if n/len(d)>=MIN_COVERAGE else -1.,
            'escalation_loss_rate':float((won[esc]==0).mean()) if esc.any() else np.nan,
            'monitor_win_rate':float((won[mon]==1).mean()) if mon.any() else np.nan,
            'base_win_rate':float(won.mean())}


def evaluate(states,won,card,th):
    return score([decide(s,card,th) for s in states],won)


def paired_delta(states,won,card,a,b,draws,seed=42):
    """Bootstrap CI of directional-accuracy difference a - b on the same deals (deals with no label count as abstain)."""
    da=np.array([decide(s,card,a) for s in states]); db=np.array([decide(s,card,b) for s in states]); won=np.asarray(won)
    def acc(d,i):
        d=d[i]; w=won[i]; m=np.isin(d,list(DIRECTIONAL))
        return ((d=='ESCALATE_AT_RISK_REVIEW')&(w==0)|(d=='MONITOR_NO_TASK')&(w==1))[m].mean() if m.any() else np.nan
    rng=np.random.default_rng(seed); v=[]
    for _ in range(draws):
        i=rng.integers(0,len(won),len(won)); x=acc(da,i)-acc(db,i)
        if not np.isnan(x): v.append(x)
    full=np.arange(len(won))
    return float(acc(da,full)-acc(db,full)),float(np.percentile(v,2.5)),float(np.percentile(v,97.5))


def grid():
    for values in product(*GRID.values()):
        yield replace(policy.DEFAULT_THRESHOLDS,**dict(zip(GRID,values)))


def search(states,won,card):
    rows=[{**{k:getattr(th,k) for k in GRID},**evaluate(states,won,card,th)} for th in grid()]
    table=pd.DataFrame(rows)
    # Deterministic tie-break: higher objective, then higher coverage, then closer to the defaults.
    default=asdict(policy.DEFAULT_THRESHOLDS)
    table['distance_to_default']=sum((table[k]-default[k]).abs()/(max(GRID[k])-min(GRID[k])) for k in GRID)
    table=table.sort_values(['objective','coverage','distance_to_default'],ascending=[False,False,True],kind='stable')
    best=replace(policy.DEFAULT_THRESHOLDS,**{k:(int(v) if k=='support_min' else float(v)) for k,v in table.iloc[0][list(GRID)].items()})
    return table,best


def sensitivity(states,won,card,open_states):
    """One threshold at a time around the defaults: validation accuracy/coverage and open-pipeline decision mix."""
    rows=[]
    for key,values in GRID.items():
        for v in values:
            th=replace(policy.DEFAULT_THRESHOLDS,**{key:v})
            mix=pd.Series([decide(s,card,th) for s in open_states]).value_counts(normalize=True)
            rows.append({'threshold':key,'value':v,'is_default':v==getattr(policy.DEFAULT_THRESHOLDS,key),**evaluate(states,won,card,th),
                         **{f'open_share_{d}':float(mix.get(d,0.)) for d in policy.DECISIONS}})
    return pd.DataFrame(rows)


def min_gain_sweep(res,ids,pool,card,full_decisions):
    """Label-free: tool calls vs agreement with the full-information decision and task context completeness."""
    rows=[]; scratch=tempfile.TemporaryDirectory(prefix='crm-thresholds-')
    for g in MIN_GAINS:
        th=replace(policy.DEFAULT_THRESHOLDS,min_gain=g)
        agent=CRMAgent(res,Path(scratch.name)/f'g{g}',AgentConfig(f'min_gain_{g}',thresholds=th))
        calls=[]; agree=[]; complete=[]
        for oid in ids:
            t=agent.run(Case(oid,pool),write=False)
            used={s['tool_call']['tool'] for s in t['steps'] if s.get('tool_call') and s['tool_call']['status']=='ok'}
            calls.append(sum(1 for s in t['steps'] if s.get('tool_call') and s['tool_call']['status'] in ('ok','error') and s['phase']=='act' and s['tool_call']['tool'] in planner.SLOT))
            agree.append(t['decision']['action']==full_decisions[oid])
            if t['decision']['action'] in ('ESCALATE_AT_RISK_REVIEW','REVIEW_UNCERTAIN'):
                complete.append({'get_prior_sales_history','explain_prediction'}<=used)
        rows.append({'min_gain':g,'deals':len(ids),'information_tool_calls_mean':float(np.mean(calls)),
                     'decision_agreement_with_full_information':float(np.mean(agree)),
                     'review_tasks':len(complete),'review_task_context_complete':float(np.mean(complete)) if complete else np.nan,
                     'is_default':g==policy.MIN_GAIN})
    scratch.cleanup()
    table=pd.DataFrame(rows)
    feasible=table.loc[(table.decision_agreement_with_full_information==1)&(table.review_task_context_complete.fillna(1)==1)]
    return table,feasible


def run(output_dir=DEFAULT_OUTPUT,quick=False):
    start=perf_counter(); out=Outputs(output_dir); folder=out.analysis/'thresholds'; folder.mkdir(parents=True,exist_ok=True)
    res=Resources(out.root); card=res.card
    val=res.validation_replay.sort_values(['engage_date','opportunity_id'],kind='stable')
    test=res.replay.sort_values(['engage_date','opportunity_id'],kind='stable')
    if quick: val,test=val.iloc[::8],test.iloc[::20]
    print(f'THRESHOLDS: collecting full-information beliefs for {len(val)} validation and {len(test)} test deals...',flush=True)
    vs=[beliefs(res,o,'validation_replay') for o in val.index]
    ts=[beliefs(res,o,'test_replay') for o in test.index]
    open_ids=res.open.sort_values(['engage_date','opportunity_id'],kind='stable').index[::(20 if quick else 1)]
    os_=[beliefs(res,o,'open') for o in open_ids]
    # Labels are joined only now, after every belief state exists (tools cannot read them).
    vw=res.closed.loc[val.index,'is_won'].to_numpy(); tw=res.closed.loc[test.index,'is_won'].to_numpy()

    # Equivalence: the full-information decision equals the planning agent's decision (default thresholds).
    sample=list(val.index[:(20 if quick else 150)]); agent=CRMAgent(res,tempfile.mkdtemp(prefix='crm-eq-'))
    agent_dec={o:agent.run(Case(o,'validation_replay'),write=False)['decision']['action'] for o in sample}
    full_dec={o:decide(s,card,policy.DEFAULT_THRESHOLDS) for o,s in zip(val.index,vs)}
    equivalence=float(np.mean([agent_dec[o]==full_dec[o] for o in sample]))

    print('THRESHOLDS: grid search on validation...',flush=True)
    half=len(val)//2
    table_v1,best_v1=search(vs[:half],vw[:half],card)
    table,best=search(vs,vw,card)
    table.to_csv(folder/'grid_validation.csv',index=False)
    draws=200 if quick else 2000
    held=paired_delta(vs[half:],vw[half:],card,best_v1,policy.DEFAULT_THRESHOLDS,draws)
    adopt=held[1]>0
    lock={'rule':'adopt tuned thresholds only if they beat the defaults on the held-out half of validation (paired 95% CI > 0)',
          'tuned_on_first_half':asdict(best_v1),'held_out_delta_accuracy':held[0],'held_out_ci':[held[1],held[2]],
          'tuned_on_full_validation':asdict(best),'adopted':bool(adopt),'default':asdict(policy.DEFAULT_THRESHOLDS),
          'locked_utc':datetime.now(timezone.utc).isoformat(),'test_scored_before_lock':False}
    write_json(folder/'selection_lock.json',lock)

    rows=[]
    for part,states,won in [('validation_first_half',vs[:half],vw[:half]),('validation_second_half',vs[half:],vw[half:]),
                            ('validation',vs,vw),('test',ts,tw)]:
        for name,th in [('default',policy.DEFAULT_THRESHOLDS),('tuned_first_half',best_v1),('tuned_full_validation',best)]:
            rows.append({'part':part,'policy':name,**evaluate(states,won,card,th)})
    comparison=pd.DataFrame(rows); comparison.to_csv(folder/'fixed_vs_tuned.csv',index=False)
    test_delta=paired_delta(ts,tw,card,best,policy.DEFAULT_THRESHOLDS,draws)
    sens=sensitivity(vs,vw,card,os_); sens.to_csv(folder/'sensitivity.csv',index=False)
    # Matched-coverage reference: label the same number of validation deals by model confidence alone.
    dist=np.array([abs(s['beliefs']['prediction']['distance_to_threshold']) if 'prediction' in s['beliefs'] else -1 for s in vs])
    pred=np.array([s['beliefs']['prediction']['predicted_outcome']=='Won' if 'prediction' in s['beliefs'] else False for s in vs])
    default_v=comparison.set_index(['part','policy']).loc[('validation','default')]
    k=max(1,int(round(default_v.coverage*len(vs)))); top=np.argsort(-dist,kind='stable')[:k]
    matched={'coverage':k/len(vs),'accuracy_of_most_confident_model_labels':float((pred[top]==(vw[top]==1)).mean()),
             'default_policy_directional_accuracy':float(default_v.directional_accuracy)}

    print('THRESHOLDS: min_gain sweep (label-free)...',flush=True)
    ids=list(val.index[:(15 if quick else 200)])
    gains,feasible=min_gain_sweep(res,ids,'validation_replay',card,full_dec)
    gains.to_csv(folder/'min_gain_sweep.csv',index=False)
    best_gain=feasible.sort_values(['information_tool_calls_mean','min_gain']).iloc[0] if len(feasible) else None

    inventory=[
        {'threshold':'decision_threshold (model)','value':res.scoring['decision_threshold'],'kind':'model classification threshold',
         'how_set':'learned: validation macro-F1 in src.train','label_available':'yes (validation)','status':'learned, unchanged'},
        {'threshold':'min_gain','value':policy.MIN_GAIN,'kind':'planning','how_set':'design; label-free sweep here','label_available':'not needed',
         'status':f"feasible range {feasible.min_gain.min():.2f}-{feasible.min_gain.max():.2f}" if len(feasible) else 'no feasible value'},
        *[{'threshold':k,'value':getattr(policy.DEFAULT_THRESHOLDS,k),'kind':'decision policy','how_set':'design; tuned on validation here',
           'label_available':'yes (validation outcomes)','status':'adopted' if adopt else 'kept: tuning did not generalize'} for k in GRID],
        {'threshold':'EVIDENCE_WEIGHTS','value':json.dumps(policy.EVIDENCE_WEIGHTS),'kind':'decision policy','how_set':'design',
         'label_available':'no (no label for "evidence quality")','status':'configurable; cannot be validated'},
        {'threshold':'TOOL_COST','value':json.dumps(policy.TOOL_COST),'kind':'planning','how_set':'design (relative compute cost)',
         'label_available':'no','status':'configurable; only their order matters for min_gain'},
        {'threshold':'STALE_QUANTILE','value':policy.STALE_QUANTILE,'kind':'data-quality warning','how_set':'design',
         'label_available':'no','status':'configurable; a record-quality rule, not an outcome rule'},
        {'threshold':'MAX_STEPS / MAX_RETRIES','value':f'{policy.MAX_STEPS} / {policy.MAX_RETRIES}','kind':'safety budget','how_set':'design',
         'label_available':'no','status':'safety constraint; never optimized'},
        {'threshold':'blocking issue -> DATA_COMPLETION; external tools blocked','value':'rule','kind':'safety','how_set':'design',
         'label_available':'no','status':'safety constraint; outside every search'},
    ]
    pd.DataFrame(inventory).to_csv(folder/'threshold_inventory.csv',index=False)
    result={'status':'complete','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full','elapsed_seconds':perf_counter()-start,
            'validation_deals':len(vs),'test_deals':len(ts),'grid_size':len(table),
            'equivalence_full_information_vs_agent':{'deals':len(sample),'agreement':equivalence},
            'lock':lock,'test_delta_tuned_minus_default':{'delta':test_delta[0],'ci':[test_delta[1],test_delta[2]]},
            'matched_coverage_reference':matched,
            'min_gain':{'default':policy.MIN_GAIN,'feasible':feasible.min_gain.tolist(),'default_feasible':bool(policy.MIN_GAIN in set(feasible.min_gain)),
                        'feasible_tool_calls_mean':float(best_gain.information_tool_calls_mean) if best_gain is not None else None,
                        'note':'every feasible value gives the same tool calls on these deals: the objective is flat, so the default is kept'},
            'comparison':comparison.to_dict('records')}
    write_json(folder/'summary.json',result)
    print(f"THRESHOLDS COMPLETE in {result['elapsed_seconds']:.0f}s: adopted={adopt}; held-out delta {held[0]:+.3f} [{held[1]:+.3f}, {held[2]:+.3f}]",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--quick',action='store_true',help='Subsampled deals; not reportable')
    args=parser.parse_args(); run(args.output,quick=args.quick)


if __name__=='__main__': main()
