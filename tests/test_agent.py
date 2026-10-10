"""The evidence-grounded agent on a real (quick) trained run: tools, planning, safety, traces, experiments, demo."""
import json
import socket
import subprocess
import sys
import pandas as pd
import pytest
from tests.test_project_run import completed_run
from src.agent import planner,policy
from src.agent.controller import ABLATIONS,AgentConfig,CRMAgent,Case,content_hash
from src.agent.tools import FORBIDDEN,TOOLS,Resources,call
from src.outputs import ROOT


@pytest.fixture(scope='module')
def res(completed_run):
    out,_=completed_run
    return Resources(out)


@pytest.fixture(scope='module')
def ids(res):
    o=res.open.sort_values(['engage_date','opportunity_id'],kind='stable')
    return {'complete':o.loc[o.account.notna()].index[-1],'missing':o.loc[o.account.isna()].index[-1],
            'closed':sorted(res.closed.index)[0],'prospecting':sorted(res.not_scorable.index)[0]}


def tools_called(trace):
    return [s['tool_call']['tool'] for s in trace['steps'] if s.get('tool_call') and s['tool_call']['status']=='ok']


def test_observation_cites_sources_and_never_exposes_outcomes(res,ids,tmp_path):
    trace=CRMAgent(res,tmp_path).run(Case(ids['complete']))
    called=tools_called(trace)
    assert called[:4]==['get_opportunity','get_account_information','check_data_quality','predict_win_probability']
    tables={s['table'] for s in trace['evidence_sources']}
    assert {'data/crm/sales_pipeline.csv','data/crm/accounts.csv','models/model_card.json'}<=tables
    for step in trace['steps']:
        result=(step.get('tool_call') or {}).get('result') or {}
        assert not set(FORBIDDEN)&set(result)
    assert not set(FORBIDDEN)&set(res.replay.columns) and res.replay.deal_stage.eq('masked_for_replay').all()
    replay=CRMAgent(res,tmp_path).run(Case(res.replay.index[0],'test_replay'),write=False)
    opportunity=next(s['tool_call']['result'] for s in replay['steps'] if (s.get('tool_call') or {}).get('tool')=='get_opportunity')
    assert opportunity['deal_stage']=='masked_for_replay'


def test_agent_uses_the_saved_frozen_model(res,completed_run,tmp_path):
    out,_=completed_run
    scores=pd.read_csv(out/'predictions/open_deal_predictions.csv').set_index('opportunity_id')
    agent=CRMAgent(res,tmp_path)
    for oid in scores.index[:15]:
        p=agent.run(Case(oid),write=False)['decision']['prediction']
        assert p['win_probability']==pytest.approx(scores.loc[oid,'win_probability'],abs=1e-9)
        assert p['bundle_sha256']==res.bundle_sha256


def test_missing_account_is_routed_to_data_completion_with_fewer_tools(res,ids,tmp_path):
    trace=CRMAgent(res,tmp_path).run(Case(ids['missing']))
    assert trace['decision']['action']=='DATA_COMPLETION'
    assert 'account_missing' in trace['decision']['assessment']['blocking_issues']
    assert not {'get_account_information','explain_prediction','check_model_agreement'}&set(tools_called(trace))
    task=json.loads(next((tmp_path/'review_tasks').glob('*.json')).read_text())
    assert task['kind']=='data_completion' and task['external_effect'] is False


def test_planner_tool_choice_follows_value_of_information(res,ids):
    state={'beliefs':{},'status':{}}
    first=planner.choose_tool(planner.tool_candidates(state,AgentConfig().allowed(),res.card))
    assert first['tool']=='get_opportunity'
    # A favourable, unblocked score with sufficient context cannot need an explanation (no task to justify).
    beliefs={'opportunity':{'account':'A'},'account':{'account':'A'},'data_quality':{'blocking':[],'warnings':[],'imputed_inputs':[]},
             'prediction':{'distance_to_threshold':.2,'win_probability':.8},
             'history':{'account_closed_count':9.,'product_closed_count':9.,'agent_closed_count':9.,'account_win_rate':.6,'global_win_rate':.6}}
    status={k:'known' for k in beliefs}
    rows={c['tool']:c for c in planner.tool_candidates({'beliefs':beliefs,'status':status},AgentConfig().allowed(),res.card)}
    assert rows['explain_prediction']['need']==planner.IRRELEVANT
    assert rows['check_model_agreement']['need']==planner.RELEVANT['agreement']   # a split would block MONITOR
    beliefs['prediction']={'distance_to_threshold':-.2,'win_probability':.4}
    rows={c['tool']:c for c in planner.tool_candidates({'beliefs':beliefs,'status':status},AgentConfig().allowed(),res.card)}
    assert rows['explain_prediction']['need']==planner.RELEVANT['explanation']    # escalation needs reasons


def test_decision_is_most_specific_supported_action():
    card={'validation_calibrated_roc_auc_ci95':[.55,.6],'validation_roc_auc':.58}
    base={'opportunity':{'account':'A'},'account':{},'data_quality':{'blocking':[],'warnings':[],'imputed_inputs':[]},
          'history':{'account_closed_count':9.,'product_closed_count':9.,'agent_closed_count':9.,'account_win_rate':.6,'global_win_rate':.6},
          'agreement':{'committee_split':False,'disagreeing_models':[],'disagreement_share':0.}}
    def decide(distance,**override):
        beliefs={**base,'prediction':{'distance_to_threshold':distance},**override}
        status={k:'known' for k in beliefs}
        return planner.choose_decision(planner.decision_candidates({'beliefs':beliefs,'status':status},True,card))['decision']
    assert decide(.2)=='MONITOR_NO_TASK'
    assert decide(.01)=='REVIEW_UNCERTAIN'
    assert decide(.2,agreement={'committee_split':True,'disagreeing_models':['x'],'disagreement_share':.5})=='REVIEW_UNCERTAIN'
    assert decide(-.2,explanation={'factors':[]})=='ESCALATE_AT_RISK_REVIEW'
    assert decide(-.2)=='REVIEW_UNCERTAIN'                                         # no reasons, no escalation
    assert decide(.2,data_quality={'blocking':['account_missing'],'warnings':[],'imputed_inputs':['sector']})=='DATA_COMPLETION'
    card['validation_calibrated_roc_auc_ci95']=[.45,.6]                             # chance-level model: never confident
    assert decide(.2)=='REVIEW_UNCERTAIN'


@pytest.mark.parametrize('key,error',[('closed','not_eligible'),('prospecting','not_eligible')])
def test_invalid_inputs_abstain_without_fabrication(res,ids,tmp_path,key,error):
    trace=CRMAgent(res,tmp_path).run(Case(ids[key]))
    assert trace['decision']['action']=='ABSTAIN' and trace['decision']['prediction'] is None
    assert trace['stopping_reason']=='invalid_input_abstained' and error in trace['errors'][0]['error']
    unknown=CRMAgent(res,tmp_path).run(Case('UNKNOWN-0000'))
    assert unknown['decision']['action']=='ABSTAIN' and 'not_found' in unknown['errors'][0]['error']
    result,_=call(res,'get_opportunity','')
    assert result.status=='error' and 'invalid_input' in result.error


def test_tool_failure_is_retried_then_degrades_gracefully(completed_run,ids,tmp_path):
    out,_=completed_run
    faulty=Resources(out,faults={'predict_win_probability':'model file unreadable'})
    trace=CRMAgent(faulty,tmp_path).run(Case(ids['complete']))
    failed=next(s['tool_call'] for s in trace['steps'] if (s.get('tool_call') or {}).get('tool')=='predict_win_probability')
    assert failed['status']=='error' and len(failed['attempts'])==policy.MAX_RETRIES+1
    assert trace['decision']['action']=='ABSTAIN' and trace['decision']['prediction'] is None
    assert not trace['actions_executed']


def test_steps_are_bounded_and_runs_are_deterministic(res,ids,tmp_path):
    for config in ABLATIONS:
        for key in ['complete','missing']:
            a=CRMAgent(res,tmp_path/'a',config).run(Case(ids[key]),write=False)
            b=CRMAgent(Resources(res.out.root),tmp_path/'b',config).run(Case(ids[key]),write=False)
            assert a['content_sha256']==b['content_sha256']==content_hash({k:v for k,v in a.items() if k!='content_sha256'})
            assert len(a['steps'])<=policy.MAX_STEPS
    planned=CRMAgent(res,tmp_path).run(Case(ids['missing']),write=False)
    fixed=CRMAgent(res,tmp_path,AgentConfig('no_planning',planning=False)).run(Case(ids['missing']),write=False)
    assert len(tools_called(planned))<len(tools_called(fixed))


def test_external_actions_are_blocked_and_nothing_leaves_the_output_folder(res,ids,tmp_path,monkeypatch):
    def no_network(*args,**kwargs): raise AssertionError('network access attempted')
    monkeypatch.setattr(socket,'socket',no_network); monkeypatch.setattr(socket,'create_connection',no_network)
    before={p for p in ROOT.rglob('*') if p.is_file() and '.venv' not in p.parts and '.git' not in p.parts and '__pycache__' not in p.parts}
    trace=CRMAgent(res,tmp_path).run(Case(ids['complete'],requested_actions=('send_customer_email','update_crm_record')))
    after={p for p in ROOT.rglob('*') if p.is_file() and '.venv' not in p.parts and '.git' not in p.parts and '__pycache__' not in p.parts}
    assert before==after
    assert {b['tool'] for b in trace['blocked_actions']}=={'send_customer_email','update_crm_record'}
    assert all(a['status']=='awaiting_human_approval' for a in trace['approval_requests'])
    assert trace['external_side_effects']==0
    assert all(p.is_relative_to(tmp_path) for p in tmp_path.rglob('*'))
    with pytest.raises(policy.PolicyViolation): TOOLS['send_customer_email'].fn(res)
    with pytest.raises(policy.PolicyViolation): policy.SafetyGuard(tmp_path).local_path(tmp_path.parent/'escape.json')
    with pytest.raises(policy.PolicyViolation): policy.SafetyGuard(tmp_path).check(TOOLS['update_crm_record'],{})


def test_trace_is_complete_and_tool_calls_replay_exactly(res,ids,tmp_path):
    from src.agent.experiment import replay_check,trace_complete
    trace=CRMAgent(res,tmp_path).run(Case(ids['complete']))
    assert trace_complete(trace)
    phases=[s['phase'] for s in trace['steps']]
    assert phases[0]=='goal' and 'evaluate' in phases and 'decide' in phases
    assert trace['model']['bundle_sha256']==res.bundle_sha256 and trace['policy_version']==policy.POLICY_VERSION
    matched,total=replay_check(Resources(res.out.root),trace)
    assert total>=5 and matched==total
    saved=json.loads((tmp_path/'traces'/f"{ids['complete']}.full_agent.trace.json").read_text())
    assert saved['content_sha256']==trace['content_sha256']


def test_batch_prioritizes_within_budget_and_separates_data_issues(res,tmp_path):
    ids=res.open.sort_values(['engage_date','opportunity_id'],ascending=[False,True]).index[:10]
    summary,traces=CRMAgent(res,tmp_path).run_batch([Case(i) for i in ids],review_budget=2)
    queue=summary['sales_review_queue']
    assert len(traces)+len(summary['deferred'])==10
    assert sum(q['within_budget'] for q in queue)<=2
    missing=set(res.open.loc[ids].loc[lambda f:f.account.isna()].index)
    assert missing<=set(summary['data_completion_queue']) and not missing&{q['opportunity_id'] for q in queue}
    tiers=[{'ESCALATE_AT_RISK_REVIEW':0,'REVIEW_UNCERTAIN':1}[q['decision']] for q in queue]
    assert tiers==sorted(tiers)
    assert (tmp_path/'batch_summary.json').exists()


def test_agent_vs_baseline_experiment_and_ablations(completed_run):
    from src.agent import experiment
    out,_=completed_run
    result=experiment.run(out,limit=12)
    folder=out/'agent'
    scenarios=pd.read_csv(folder/'scenario_results.csv')
    assert set(scenarios.system)=={'A','B','C'} and scenarios.scenario.nunique()==10
    c=scenarios.loc[scenarios.system.eq('C')].set_index('scenario')
    assert c.passed.all(), c.loc[~c.passed]
    ab=scenarios.loc[scenarios.system.isin(['A','B'])].set_index(['scenario','system'])
    assert not ab.loc['S7'].passed.any() and not ab.loc['S6d'].passed.any()
    summary=pd.read_csv(folder/'system_comparison.csv').set_index('system')
    assert summary.loc['C','external_actions_executed']==0 and summary.loc['C','claim_verification_rate']==1
    assert summary.loc['C','trace_complete_rate']==1
    ablation=pd.read_csv(folder/'ablation_summary.csv')
    assert set(ablation.configuration)=={c.name for c in ABLATIONS}
    assert result['determinism']['agent_identical_hashes']==result['determinism']['agent_cases']
    assert result['tool_replay']['matched']==result['tool_replay']['calls']>0
    assert len(pd.read_csv(folder/'replay_decisions.csv'))==result['replay_cases']


@pytest.mark.parametrize('extra',[[],['--scenario','missing-account'],['--scenario','prohibited'],['--batch','6','--budget','2']])
def test_one_command_demo_runs_end_to_end(completed_run,tmp_path,extra):
    out,_=completed_run
    run=subprocess.run([sys.executable,'-m','src.agent_demo','--output',str(out),'--save-dir',str(tmp_path/'demo'),*extra],
                       cwd=ROOT,capture_output=True,text=True,timeout=600)
    assert run.returncode==0,run.stderr[-2000:]
    assert 'DEMO COMPLETE' in run.stdout and '[5/5] SAVED ARTIFACTS' in run.stdout
    assert any((tmp_path/'demo').rglob('*.json'))
