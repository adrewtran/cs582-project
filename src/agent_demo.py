"""One-command live demo of the evidence-grounded CRM agent on real saved models and real CRM data.

    python -m src.agent_demo                         # one reproducible real Engaging deal, narrated step by step
    python -m src.agent_demo --scenario missing-account
    python -m src.agent_demo --scenario borderline | conflict | invalid | tool-failure | prohibited
    python -m src.agent_demo --opportunity-id 0BEK2NIL
    python -m src.agent_demo --batch 10 --budget 3   # several deals: autonomous review prioritization
    python -m src.agent_demo --pause                 # wait for Enter between phases (live presentation)
    python -m src.agent_demo --check                 # verify the saved models and results, then exit
    python -m src.agent_demo --traditional           # Baseline A only: the ordinary ML prediction for the demo deal

Offline and deterministic: no API key, no network, no training. It needs the saved run in reports/crm/final/
(created by python -m src.run_project). Results are written to reports/crm/agent_demo/ (git-ignored).
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import textwrap
import pandas as pd
from src.agent import policy
from src.agent.controller import CRMAgent,Case
from src.agent.experiment import baseline_a,baseline_b
from src.agent.scenarios import REVIEW_BUDGET,select
from src.agent.tools import Resources
from src.outputs import DEFAULT_OUTPUT,ROOT

DEMO_OUTPUT=ROOT/'reports/crm/agent_demo'
SCENARIOS={'complete':'S1','missing-account':'S2','borderline':'S3','conflict':'S4','invalid':'S6b','tool-failure':'S6d','prohibited':'S7'}
WIDTH=96


def say(text='',indent=0):
    prefix=' '*indent
    for line in (text.splitlines() or ['']):
        print(textwrap.fill(line,WIDTH,initial_indent=prefix,subsequent_indent=prefix+'  ') if line else '',flush=True)


def banner(title,pause=False):
    if pause and sys.stdin.isatty(): input('\n[Enter] to continue... ')
    print('\n'+'='*WIDTH+f'\n{title}\n'+'='*WIDTH,flush=True)


def short(value,limit=90):
    text=json.dumps(value,ensure_ascii=False) if not isinstance(value,str) else value
    return text if len(text)<=limit else text[:limit-3]+'...'


def narrate(step):
    """Live view of every trace step as the agent produces it."""
    phase=step['phase'].upper(); chosen=step.get('chosen')
    if step['phase']=='goal':
        say(f"Step {step['step']:>2} GOAL     {step['goal']['description']}"); return
    if step['phase']=='plan' and chosen is None:
        say(f"Step {step['step']:>2} PLAN     {step['evaluation']}")
        for c in step.get('candidates',[]):
            if c['applicable']: say(f"skip {c['tool']}: need {c['need']:.2f} - cost {c['cost']:.2f} = {c['utility']:.2f} < {policy.MIN_GAIN} ({c['reason']})",14)
        return
    if step['phase']=='decide':
        say(f"Step {step['step']:>2} DECIDE   candidates, most specific first:")
        for c in step['candidates']:
            failed=[k for k,v in c['requirements'].items() if not v]
            mark='CHOSEN ' if c['decision']==chosen else ('valid  ' if c['valid'] else 'blocked')
            say(f"[{mark}] {c['decision']}"+(f"  (missing: {', '.join(failed)})" if failed else ''),14)
        return
    call=step.get('tool_call') or {}
    applicable=sorted([c for c in step.get('candidates',[]) if c.get('applicable')],key=lambda c:-c['utility'])
    if applicable:
        say(f"Step {step['step']:>2} PLAN     options (need - cost): "+'; '.join(f"{c['tool']} {c['utility']:.2f}" for c in applicable[:4]))
        say(f"why {chosen}: {next(c['reason'] for c in applicable if c['tool']==chosen)}",14)
    label='ACT' if step['phase']=='act' else phase
    args=short(call.get('arguments') or {},60)
    say(f"{'' if applicable else f'Step {step['step']:>2} '}{'' if not applicable else ' '*8}{label:<8} {call.get('tool',chosen)}({args}) -> {call.get('status','').upper()}")
    if call.get('sources'): say('source: '+'; '.join(f"{s['table']} [{s['key']}]" for s in call['sources'][:2]),14)
    say('result: '+step.get('evaluation',''),14)


def compare(res,case,trace):
    banner('COMPARISON ON THE SAME DEAL: traditional ML (A) vs rule assistant (B) vs agent (C)')
    try:
        a=baseline_a(res,case)
        say(f"A  Prediction only : P(Won) {a['win_probability']:.4f} -> predicted {a['predicted_outcome']}, priority {a['priority']}")
        say(f"top factor: {a['positive_factors'].split('; ')[0]} | warning: {a['input_warning'] or 'none'}",21)
    except Exception as exc: say(f'A  Prediction only : no output ({type(exc).__name__}: {exc})')
    try:
        b=baseline_b(res,case); rec=b['agent_recommendation']
        say(f"B  Rule assistant  : loss risk {rec['loss_risk']}; fixed rules -> {', '.join(x['rule_id'] for x in rec['actions'])}")
        say('prints advice only; creates no task, keeps no trace, cannot refuse an external request',21)
    except Exception as exc: say(f'B  Rule assistant  : no output ({type(exc).__name__}: {exc})')
    d=trace['decision']; tools=[s['tool_call']['tool'] for s in trace['steps'] if s.get('tool_call') and s['tool_call']['status']=='ok']
    say(f"C  Agent           : {d['action']} after {len(tools)} tool calls chosen by need; {trace['final_recommendation']}")
    say(f"evidence sources {len(trace['evidence_sources'])}; tasks {[a['task_id'] for a in trace['actions_executed']]}; "
        f"blocked external requests {len(trace['blocked_actions'])}; trace hash {trace['content_sha256'][:12]}",21)


def single(res,args,case,folder,title):
    banner(f'[2/5] THE AGENT STARTS: {title}',args.pause)
    say(f'Opportunity {case.opportunity_id}; requested actions: {list(case.requested_actions) or "none"}; step budget {policy.MAX_STEPS}')
    agent=CRMAgent(res,folder,verbose=None if args.quiet else narrate)
    trace=agent.run(case)
    banner('[3/5] DECISION AND ACTIONS',args.pause)
    say(f"Decision: {trace['decision']['action']}")
    say(f"Recommendation: {trace['final_recommendation']}")
    a=trace['decision'].get('assessment')
    if a:
        say(f"Evidence quality {a['evidence_quality']:.2f} (needs {policy.SUFFICIENT_EVIDENCE}); risk signal {a['risk_signal']}; "
            f"model reliability {a['model_reliability']} (validation ROC-AUC {a['validation_roc_auc']:.3f}, 95% CI {a['validation_roc_auc_ci95'][0]:.3f}-{a['validation_roc_auc_ci95'][1]:.3f})")
        if a['conflicts']: say('Conflicts: '+short([c['code'] for c in a['conflicts']]))
    for x in trace['actions_executed']: say(f"Executed {x['tool']}: {x['status']} (task {x['task_id']})")
    for x in trace['blocked_actions']: say(f"BLOCKED {x['tool']}: {x['reason']}")
    for e in trace['errors']: say(f"Error recorded: {e['tool']}: {e['error']}")
    say(f"Stopping reason: {trace['stopping_reason']}; external side effects: {trace['external_side_effects']}")
    compare(res,case,trace)
    return trace


def check(run_dir):
    """Step 1 of the demo guide: are the trained models and saved results present and consistent?"""
    import hashlib
    from src.outputs import Outputs
    out=Outputs(run_dir); ok=True
    banner('CHECKING SAVED MODELS AND RESULTS (read-only)')
    required=['run_manifest.json','models/model_bundle.joblib','models/trained_models.joblib','models/model_card.json',
              'models/selection_lock.json','metrics/test_metrics.csv','metrics/feature_set_comparison.csv',
              'predictions/open_deal_predictions.csv','agent/experiment.json','agent/system_comparison.csv']
    for name in required:
        path=out.root/name; present=path.exists(); ok&=present
        say(f"[{'OK' if present else 'MISSING'}] {name}"+(f'  ({path.stat().st_size:,} bytes)' if present else ''))
    if not ok:
        say('Some files are missing. Run: python -m src.run_project   (about 10-15 minutes on a laptop CPU)'); sys.exit(1)
    m=out.read_manifest(); lock=json.loads((out.models/'selection_lock.json').read_text())
    test=pd.read_csv(out.metrics/'test_metrics.csv').set_index('model')
    say(f"Run status: {m['status']} | mode: {m['mode']} | finished: {m.get('completed_utc','?')[:19]} | platform: {m.get('platform','?')}")
    say(f"Selected on validation BEFORE test: {lock['selected_feature_set']}/{lock['selected_model']} "
        f"(test_evaluated_at_selection={lock['test_evaluated_at_selection']})")
    say(f"Selected model test ROC-AUC {test.loc[m['selected_model'],'roc_auc']:.4f}; accuracy {test.loc[m['selected_model'],'accuracy']:.4f}")
    say(f"Scored open deals: {m['scored_open_rows']:,}; bundle sha256 {hashlib.sha256((out.models/'model_bundle.joblib').read_bytes()).hexdigest()[:16]}...")
    e=json.loads((out.root/'agent/experiment.json').read_text())
    say(f"Agent experiment: {e['status']}, {e['open_cases']:,} open deals, determinism {e['determinism']['agent_identical_hashes']}/{e['determinism']['agent_cases']}, "
        f"tool replay {e['tool_replay']['matched']}/{e['tool_replay']['calls']}")
    print('CHECK PASSED',flush=True)


def traditional(res,args):
    """Step 2 of the demo guide: what an ordinary ML pipeline gives for the same reproducible deal."""
    scores=pd.read_csv(res.out.predictions/'open_deal_predictions.csv')
    oid=args.opportunity_id or next(s for s in select(res,scores) if s.id=='S1').opportunity_ids[0]
    banner(f'TRADITIONAL ML PREDICTION (Baseline A) for {oid}')
    a=baseline_a(res,Case(oid))
    say(f"Deal {oid}: {a['product']}, agent {a['sales_agent']}, account {a['account'] if isinstance(a['account'],str) else 'MISSING'}")
    say(f"P(Won) {a['win_probability']:.4f}   P(Lost) {a['loss_probability']:.4f}   threshold {a['decision_threshold']:.4f}")
    say(f"Predicted outcome: {a['predicted_outcome']}   priority band: {a['priority']}")
    say(f"Raises the score: {a['positive_factors']}")
    say(f"Lowers the score: {a['negative_factors']}")
    say(f"Warning: {a['input_warning'] or 'none'}")
    say('This is where a traditional pipeline stops: a number and its factors. A person must still look up the account, '
        'check history and data quality, decide what to do and record why.')
    print('DEMO COMPLETE',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0],formatter_class=argparse.RawDescriptionHelpFormatter,epilog=__doc__)
    parser.add_argument('--scenario',choices=sorted(SCENARIOS))
    parser.add_argument('--opportunity-id')
    parser.add_argument('--batch',type=int,default=0,help='Run the agent on this many recent Engaging deals')
    parser.add_argument('--budget',type=int,default=REVIEW_BUDGET,help='Maximum human review tasks in batch mode')
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT,help='Saved run folder (default reports/crm/final)')
    parser.add_argument('--save-dir',type=Path,default=DEMO_OUTPUT)
    parser.add_argument('--pause',action='store_true',help='Wait for Enter between phases')
    parser.add_argument('--quiet',action='store_true',help='Do not print every step')
    parser.add_argument('--check',action='store_true',help='Verify saved models and results, then exit')
    parser.add_argument('--traditional',action='store_true',help='Show only the ordinary ML prediction (Baseline A)')
    args=parser.parse_args()
    if args.check: return check(args.output)

    banner('[1/5] LOADING THE FROZEN, ALREADY-TRAINED SYSTEM (no training, no network)')
    res=Resources(args.output)
    m=res.manifest; card=res.card
    if m.get('mode')!='full': say(f"WARNING: {args.output} is a {m.get('mode')} run, not the reportable full run.")
    say(f"Run folder: {args.output}  (status {m.get('status')}, platform {m.get('platform','?')})")
    say(f"Model: {res.scoring['feature_set']}/{res.scoring['model'].name}; bundle sha256 {res.bundle_sha256[:16]}...")
    say(f"Validation ROC-AUC {card['validation_roc_auc']:.3f} (95% CI {card['validation_calibrated_roc_auc_ci95'][0]:.3f}-{card['validation_calibrated_roc_auc_ci95'][1]:.3f}); "
        f"decision threshold {res.scoring['decision_threshold']:.4f}; model available {res.scoring['model_available_date']}")
    say(f"Data: {len(res.open):,} Engaging deals ({int(res.open.account.isna().sum()):,} without account); snapshot {res.snapshot_date.date()}")
    if args.traditional: return traditional(res,args)
    folder=args.save_dir.resolve()
    if folder.exists(): shutil.rmtree(folder)
    folder.mkdir(parents=True)

    if args.batch:
        ids=res.open.sort_values(['engage_date','opportunity_id'],ascending=[False,True],kind='stable').head(args.batch).opportunity_id.tolist()
        banner(f'[2/5] BATCH GOAL: which of {len(ids)} recent Engaging deals need human review first? (budget {args.budget})',args.pause)
        agent=CRMAgent(res,folder,verbose=None)
        summary,traces=agent.run_batch([Case(i,'open',(),'demo_batch') for i in ids],review_budget=args.budget)
        banner('[3/5] WHAT THE AGENT DID FOR EACH DEAL',args.pause)
        for t in sorted(traces,key=lambda t:t['case']['opportunity_id']):
            p=t['decision'].get('prediction'); a=t['decision'].get('assessment') or {}
            tools=sum(1 for s in t['steps'] if s.get('tool_call') and s['tool_call']['status']=='ok')
            say(f"{t['case']['opportunity_id']}: {t['decision']['action']:<24} P(Won) {p['win_probability'] if p else float('nan'):.3f}  "
                f"evidence {a.get('evidence_quality',0):.2f}  tools {tools}")
        for d in summary['deferred']: say(f"{d['opportunity_id']}: deferred ({d['reason']})")
        banner('[4/5] REVIEW QUEUE (triage key: escalation > uncertain; evidence quality; lower P(Won); longer open)',args.pause)
        for q in summary['sales_review_queue']:
            say(f"#{q['rank']} {q['opportunity_id']} {q['decision']:<24} P(Won) {q['win_probability']:.3f} evidence {q['evidence_quality']:.2f}"
                f"  {'TASK CREATED' if q['within_budget'] else 'over budget: listed, no task'}")
        say(f"Data-completion queue (record must be fixed first): {summary['data_completion_queue']}")
        say(f"Monitor only (no task): {summary['monitor_only']}")
        say(f"Decision counts: {summary['decision_counts']}")
        trace_path=folder/'batch_summary.json'
    else:
        scenario=None
        if args.scenario:
            scores=pd.read_csv(res.out.predictions/'open_deal_predictions.csv')
            scenario=next(s for s in select(res,scores) if s.id==SCENARIOS[args.scenario])
            if scenario.faults: res=Resources(args.output,faults=scenario.faults)
            case=Case(scenario.opportunity_ids[0],scenario.pool,scenario.requested_actions,scenario.id)
            title=f'{scenario.id} {scenario.title} ({scenario.kind}; rule: {scenario.rule})'
        else:
            if args.opportunity_id: oid=args.opportunity_id; title=f'user-selected {oid}'
            else:
                scores=pd.read_csv(res.out.predictions/'open_deal_predictions.csv')
                s1=next(s for s in select(res,scores) if s.id=='S1'); oid=s1.opportunity_ids[0]; title=f'reproducible deal: {s1.rule}'
            case=Case(oid,'open',(),'demo')
        trace=single(res,args,case,folder,title)
        trace_path=Path(trace.get('files',{}).get('trace',''))
    banner('[5/5] SAVED ARTIFACTS',args.pause)
    for path in sorted(p for p in folder.rglob('*') if p.is_file()):
        say(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path))
    say(f'Open the decision trace: {trace_path.relative_to(ROOT) if trace_path.is_relative_to(ROOT) else trace_path}')
    say('Nothing was sent and no CRM record was changed. Scores are predictive associations, not causes.')
    print('DEMO COMPLETE',flush=True)


if __name__=='__main__': main()
