"""Bounded, deterministic, tool-using CRM decision agent.

    observe -> plan (choose a tool by information need minus cost) -> act (call the tool through the guard)
    -> evaluate (update beliefs) -> ... -> assess evidence -> decide (most specific supported action)
    -> act (local review task) -> report (trace) -> stop

The agent executes read, compute and local-write tools on its own. External actions (email, CRM updates,
third parties) are never executed: the guard blocks them and the agent records an approval request.
"""
from dataclasses import dataclass,field
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import pandas as pd
from src.agent import planner,policy
from src.agent.tools import INFORMATION_TOOLS,TOOLS,call,jsonable

VOLATILE={'timestamp','duration_ms','started_utc','finished_utc','created_utc','path','trace','report','output_dir','outbox'}


@dataclass(frozen=True)
class AgentConfig:
    """Ablation switches. The default is the full proposed agent."""
    name: str='full_agent'
    evidence_retrieval: bool=True   # account, prior history and explanation tools
    reliability_checks: bool=True   # data quality, committee agreement and evidence gating
    planning: bool=True             # False: run every available tool in a fixed order
    tool_execution: bool=True       # False: decide, but write no task and no report (dry run)

    def allowed(self):
        tools=set(INFORMATION_TOOLS)
        if not self.evidence_retrieval: tools-={'get_account_information','get_prior_sales_history','explain_prediction'}
        if not self.reliability_checks: tools-={'check_data_quality','check_model_agreement'}
        return tools


ABLATIONS=[AgentConfig(),AgentConfig('no_evidence_retrieval',evidence_retrieval=False),
           AgentConfig('no_reliability_checks',reliability_checks=False),AgentConfig('no_planning',planning=False),
           AgentConfig('no_tool_execution',tool_execution=False)]


def content_hash(value):
    """Hash of a trace without timestamps, durations or absolute paths: equal runs give equal hashes."""
    def strip(v):
        if isinstance(v,dict): return {k:strip(x) for k,x in v.items() if k not in VOLATILE}
        if isinstance(v,list): return [strip(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(strip(value),sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def now(): return datetime.now(timezone.utc).isoformat()


SUGGESTED_CHECKS={
    'ESCALATE_AT_RISK_REVIEW':['Confirm the deal is still active and who owns the next step.',
                               'Review the listed negative factors and prior outcomes with the account owner.',
                               'Decide on any customer contact yourself; the agent sends nothing.'],
    'REVIEW_UNCERTAIN':['Treat the score as inconclusive; the listed evidence is mixed or incomplete.',
                        'Gather the missing context (activity, buyer intent, budget) that the CRM does not record.'],
    'DATA_COMPLETION':['Complete or correct the CRM record (see the blocking issues).',
                       'Re-run the agent after the record is fixed; until then the score is out of training support.'],
}


@dataclass
class Case:
    opportunity_id: str
    pool: str='open'
    requested_actions: tuple=()      # e.g. ('send_customer_email',) -> blocked and escalated, never executed
    label: str=''


@dataclass
class Episode:
    case: Case
    state: dict=field(default_factory=lambda:{'beliefs':{},'status':{}})
    steps: list=field(default_factory=list)
    sources: list=field(default_factory=list)
    errors: list=field(default_factory=list)


class CRMAgent:
    def __init__(self,resources,output_dir,config=AgentConfig(),verbose=None):
        self.res=resources; self.config=config
        self.output_dir=Path(output_dir).resolve()
        self.guard=policy.SafetyGuard(self.output_dir)
        self.outbox=self.output_dir/'review_tasks'; self.traces=self.output_dir/'traces'
        self.verbose=verbose   # callable(event dict) for live demo printing

    # ------------------------------------------------------------------ helpers
    def emit(self,event):
        if self.verbose: self.verbose(event)

    def record(self,ep,phase,**content):
        step={'step':len(ep.steps)+1,'timestamp':now(),'phase':phase,
              'state_before':dict(sorted(ep.state['status'].items())),**content}
        ep.steps.append(jsonable(step)); self.emit(ep.steps[-1])

    def model_info(self):
        card=self.res.card
        return {'name':self.res.scoring['model'].name,'feature_set':self.res.scoring['feature_set'],
                'bundle_sha256':self.res.bundle_sha256,'decision_threshold':self.res.scoring['decision_threshold'],
                'validation_roc_auc':card['validation_roc_auc'],'validation_roc_auc_ci95':card['validation_calibrated_roc_auc_ci95'],
                'model_available_date':self.res.scoring['model_available_date']}

    def arguments(self,ep,name):
        c=ep.case; b=ep.state['beliefs']
        if name=='get_opportunity': return {'opportunity_id':c.opportunity_id,'pool':c.pool}
        if name=='get_account_information': return {'account':b['opportunity']['account']}
        if name=='check_data_quality':
            found={'known':True,'failed':False}.get(ep.state['status'].get('account'))
            return {'opportunity_id':c.opportunity_id,'pool':c.pool,'account_found':found}
        return {'opportunity_id':c.opportunity_id,'pool':c.pool}

    def execute(self,ep,name,**arguments):
        """Run a tool through the guard; retry once on failure; never invent a result."""
        tool=TOOLS[name]; self.guard.check(tool,arguments)
        attempts=[]
        for attempt in range(policy.MAX_RETRIES+1):
            result,ms=call(self.res,name,**arguments)
            attempts.append({'attempt':attempt+1,'status':result.status,'error':result.error,'duration_ms':ms})
            if result.status=='ok': break
        return result,attempts

    # ------------------------------------------------------------------ the loop
    def observe_and_plan(self,ep,allowed,goal):
        state=ep.state
        while len(ep.steps)<policy.MAX_STEPS-3:          # keep room for assess, decide, act/report
            candidates=planner.tool_candidates(state,allowed,self.res.card,self.config.reliability_checks,fixed_sequence=not self.config.planning)
            chosen=planner.choose_tool(candidates)
            if chosen is None:
                self.record(ep,'plan',goal=goal,candidates=candidates,chosen=None,
                            evaluation='No remaining tool clears the minimum information gain; assess the evidence.')
                return 'information_sufficient'
            name=chosen['tool']; args=self.arguments(ep,name)
            result,attempts=self.execute(ep,name,**args)
            slot=chosen['slot']
            if result.status=='ok':
                state['beliefs'][slot]=jsonable(result.data); state['status'][slot]='known'
                ep.sources.extend(result.sources)
                evaluation=self.evaluate_step(slot,result.data)
            else:
                state['status'][slot]='failed'; ep.errors.append({'tool':name,'error':result.error})
                evaluation=f'{name} failed after {len(attempts)} attempt(s): {result.error}. No value is assumed.'
            self.record(ep,'act',goal=goal,candidates=candidates,chosen=name,
                        tool_call={'tool':name,'permission':TOOLS[name].permission,'arguments':args,'status':result.status,
                                   'attempts':attempts,'result':result.data,'sources':result.sources,'error':result.error},
                        evaluation=evaluation)
            if slot=='opportunity' and result.status!='ok': return 'invalid_input'
        return 'step_budget_exhausted'

    @staticmethod
    def evaluate_step(slot,data):
        if slot=='opportunity': return f"Observed {data['opportunity_id']} ({data['product']}, agent {data['sales_agent']}, account {data['account'] or 'MISSING'})."
        if slot=='account': return f"Account found: {data['account']} ({data['sector']}, {data['employees']} employees)."
        if slot=='data_quality':
            return f"Blocking issues: {data['blocking'] or 'none'}; warnings: {data['warnings'] or 'none'}."
        if slot=='prediction': return f"Calibrated P(Won) {data['win_probability']:.4f} vs threshold {data['decision_threshold']:.4f}: {planner.risk_signal(data)}."
        if slot=='history': return (f"Prior closed deals: agent {data['agent_closed_count']:.0f}, account {data['account_closed_count']:.0f}, "
                                    f"product {data['product_closed_count']:.0f} (archive win rate {data['global_win_rate']:.3f}).")
        if slot=='agreement': return f"Committee disagreement {data['disagreement_share']:.2f} ({'split' if data['committee_split'] else 'agrees'})."
        if slot=='explanation': return 'Top factor: '+(f"{data['factors'][0]['feature']} ({data['factors'][0]['delta_p_win']:+.4f})" if data['factors'] else 'none')
        return ''

    def assess(self,ep,goal):
        if ep.state['status'].get('opportunity')!='known': return
        result,attempts=self.execute(ep,'evaluate_evidence',beliefs=ep.state['beliefs'])
        if result.status=='ok':
            ep.state['beliefs']['assessment']=jsonable(result.data); ep.state['status']['assessment']='known'
            ep.sources.extend(result.sources)
            a=result.data
            evaluation=(f"Evidence quality {a['evidence_quality']:.2f} ({'sufficient' if a['sufficient'] else 'insufficient'}); "
                        f"risk {a['risk_signal']}; conflicts {[c['code'] for c in a['conflicts']] or 'none'}; model reliability {a['model_reliability']}.")
        else:
            ep.state['status']['assessment']='failed'; evaluation=f'evaluate_evidence failed: {result.error}'
        self.record(ep,'evaluate',goal=goal,chosen='evaluate_evidence',
                    tool_call={'tool':'evaluate_evidence','permission':'compute','arguments':{'beliefs':'current belief state'},
                               'status':result.status,'attempts':attempts,'result':result.data,'sources':result.sources,'error':result.error},
                    evaluation=evaluation)

    def decide(self,ep,goal):
        candidates=planner.decision_candidates(ep.state,self.config.reliability_checks,self.res.card)
        chosen=planner.choose_decision(candidates)
        decision=chosen['decision']
        reasons=[f"{c['decision']} not supported: "+', '.join(k for k,v in c['requirements'].items() if not v)
                 for c in candidates[:candidates.index(chosen)] if c['requirements']]
        self.record(ep,'decide',goal=goal,candidates=candidates,chosen=decision,
                    evaluation=f'Most specific supported action: {decision}.',rejected=reasons)
        return decision,reasons

    def task_for(self,ep,decision,reasons):
        b=ep.state['beliefs']; a=b.get('assessment') or {}; p=b.get('prediction') or {}
        kind='data_completion' if decision=='DATA_COMPLETION' else 'sales_review'
        priority={'ESCALATE_AT_RISK_REVIEW':'high','REVIEW_UNCERTAIN':'medium','DATA_COMPLETION':'normal'}[decision]
        evidence={'opportunity':b.get('opportunity'),'account':b.get('account'),
                  'data_quality':(b.get('data_quality') or {}).get('issues'),
                  'win_probability':p.get('win_probability'),'decision_threshold':p.get('decision_threshold'),
                  'risk_signal':a.get('risk_signal'),'evidence_quality':a.get('evidence_quality'),
                  'conflicts':a.get('conflicts'),'model_reliability':a.get('model_reliability'),
                  'prior_history':{k:v for k,v in (b.get('history') or {}).items() if k.endswith(('closed_count','win_rate'))},
                  'top_factors':(b.get('explanation') or {}).get('factors',[])[:3]}
        task_id=hashlib.sha256(f'{policy.POLICY_VERSION}|{ep.case.opportunity_id}|{decision}'.encode()).hexdigest()[:16]
        return jsonable({'task_id':task_id,'kind':kind,'priority':priority,'opportunity_id':ep.case.opportunity_id,
                'decision':decision,'why_not_more_specific':reasons,'suggested_human_checks':SUGGESTED_CHECKS[decision],
                'evidence':evidence,'sources':ep.sources,'model':self.model_info(),'created_by':'agent',
                'created_utc':now(),'status':'open','external_effect':False,
                'note':'Predictive association only; not a causal or revenue claim. A human decides any customer action.'})

    def act(self,ep,decision,reasons,goal,write=True):
        executed=[]; blocked=[]; approvals=[]
        if decision in SUGGESTED_CHECKS:
            task=self.task_for(ep,decision,reasons)
            if self.config.tool_execution and write:
                result,attempts=self.execute(ep,'create_review_task',guard=self.guard,outbox=self.outbox,task=task)
                status=result.status; data=result.data
            else:
                status='not_executed'; attempts=[]; data={'task_id':task['task_id'],'dry_run':True}
            executed.append({'tool':'create_review_task','status':status,'task_id':task['task_id']})
            self.record(ep,'act',goal=goal,chosen='create_review_task',
                        tool_call={'tool':'create_review_task','permission':'local_write','arguments':{'task_id':task['task_id'],'kind':task['kind']},
                                   'status':status,'attempts':attempts,'result':data,'sources':[],'error':''},
                        evaluation=f"{task['kind']} task {task['task_id']} ({task['priority']} priority): {status}.")
        for name in ep.case.requested_actions:
            tool=TOOLS.get(name)
            try:
                if tool is None: raise policy.PolicyViolation(f'{name} is not a registered tool')
                self.guard.check(tool,{})
                status='allowed'   # no registered external tool is autonomous; kept for completeness
            except policy.PolicyViolation as exc:
                status='blocked'; blocked.append({'tool':name,'reason':str(exc)})
                approvals.append({'tool':name,'opportunity_id':ep.case.opportunity_id,'status':'awaiting_human_approval',
                                  'agent_position':'not executed; a human must decide after reviewing the task'})
            self.record(ep,'act',goal=goal,chosen=name,
                        tool_call={'tool':name,'permission':tool.permission if tool else 'unknown','arguments':{},
                                   'status':status,'attempts':[],'result':{},'sources':[],'error':blocked[-1]['reason'] if blocked else ''},
                        evaluation='External action requested: blocked by the safety guard and queued for human approval.')
        return executed,blocked,approvals

    def recommendation(self,ep,decision):
        b=ep.state['beliefs']; a=b.get('assessment') or {}; p=b.get('prediction')
        score=f"P(Won) {p['win_probability']:.3f}" if p else 'no score'
        why=[]
        if a.get('risk_signal')=='borderline': why.append(f'score within {policy.BORDERLINE_MARGIN} of the decision threshold')
        why+=[c['code'].replace('_',' ') for c in a.get('conflicts',[])]
        if a and not a.get('sufficient'): why.append(f"evidence quality {a['evidence_quality']:.2f} below {policy.SUFFICIENT_EVIDENCE}")
        if a.get('model_reliability')=='unusable': why.append('model not distinguishable from chance on validation')
        if decision=='REVIEW_UNCERTAIN' and ep.state['status'].get('explanation')!='known' and a.get('risk_signal')=='at_risk':
            why.append('no explanation available for an escalation')
        blocking=', '.join(a.get('blocking_issues',[])) or 'see errors'
        return {'ESCALATE_AT_RISK_REVIEW':f'Human review first: {score} is below the threshold, with sufficient evidence ({a.get("evidence_quality",0):.2f}) and no conflicts.',
                'MONITOR_NO_TASK':f'No task: {score} is favourable, with sufficient evidence and no conflicts; keep monitoring.',
                'REVIEW_UNCERTAIN':f"Human review, low confidence: {score}; {'; '.join(why) or 'evidence is mixed'}.",
                'DATA_COMPLETION':f'Fix the CRM record first ({blocking}): the score is outside training support until then.',
                'ABSTAIN':'No recommendation: required information could not be obtained (see errors).'}[decision]

    def markdown(self,trace):
        d=trace['decision']; lines=[f"# Agent decision: {trace['case']['opportunity_id']}",'',
            f"- Goal: {trace['goal']['description']}",f"- Decision: **{d['action']}**",f"- Recommendation: {trace['final_recommendation']}",
            f"- Model: {trace['model']['feature_set']}/{trace['model']['name']} (validation ROC-AUC {trace['model']['validation_roc_auc']:.3f})",
            f"- Stopping reason: {trace['stopping_reason']}; steps {len(trace['steps'])}/{policy.MAX_STEPS}",
            f"- Trace hash (content): `{trace['content_sha256'][:16]}`",'','## Steps','','| # | Phase | Chosen | Status | Evaluation |','|---|---|---|---|---|']
        for s in trace['steps']:
            status=(s.get('tool_call') or {}).get('status','')
            lines.append(f"| {s['step']} | {s['phase']} | {s.get('chosen') or '-'} | {status} | {s.get('evaluation','').replace('|','/')} |")
        if trace['blocked_actions']:
            lines+=['','## Blocked external actions','']+[f"- {b['tool']}: {b['reason']}" for b in trace['blocked_actions']]
        lines+=['','Predictive associations only. No customer was contacted and no CRM record was changed.','']
        return '\n'.join(lines)

    def run(self,case,prior=None,write=True,goal=None,allowed=None,stop_after_screen=False):
        """Run one bounded episode. ``prior`` continues from a screening episode in batch mode."""
        started=now()
        ep=prior or Episode(case)
        goal=goal or {'id':'TRIAGE_OPPORTUNITY','description':f'Decide the safest supported review action for {case.opportunity_id}'}
        allowed=self.config.allowed() if allowed is None else allowed
        if not ep.steps: self.record(ep,'goal',goal=goal,evaluation='Goal selected; start by observing the opportunity.')
        stop=self.observe_and_plan(ep,allowed,goal)
        if stop_after_screen: return ep
        if stop!='invalid_input': self.assess(ep,goal)
        decision,reasons=self.decide(ep,goal)
        executed,blocked,approvals=self.act(ep,decision,reasons,goal,write=write)
        stopping={'invalid_input':'invalid_input_abstained','step_budget_exhausted':'step_budget_exhausted'}.get(stop,
                  'goal_reached_decision_made' if decision!='ABSTAIN' else 'required_information_unavailable')
        trace={'trace_version':1,'agent':'EvidenceGroundedCRMAgent','policy_version':policy.POLICY_VERSION,
               'configuration':self.config.name,'case':{'opportunity_id':case.opportunity_id,'pool':case.pool,'label':case.label,
               'requested_actions':list(case.requested_actions)},'goal':goal,'model':self.model_info(),
               'started_utc':started,'finished_utc':now(),'steps':ep.steps,
               'decision':{'action':decision,'why_not_more_specific':reasons,
                           'assessment':ep.state['beliefs'].get('assessment'),'prediction':ep.state['beliefs'].get('prediction')},
               'final_recommendation':self.recommendation(ep,decision),'actions_executed':executed,
               'blocked_actions':blocked,'approval_requests':approvals,'errors':ep.errors,
               'evidence_sources':ep.sources,'stopping_reason':stopping,'step_budget':policy.MAX_STEPS,
               'external_side_effects':0}
        trace=jsonable(trace); trace['content_sha256']=content_hash(trace)
        if self.config.tool_execution and write:
            name=f"{case.opportunity_id}.{self.config.name}"
            result,_=self.execute(ep,'export_agent_report',guard=self.guard,folder=self.traces,stem=name,trace=trace,markdown=self.markdown(trace))
            trace['files']=result.data
        return trace

    # ------------------------------------------------------------------ batch
    def run_batch(self,cases,review_budget=5,investigate_budget=None,write=True):
        """Goal: a human review queue of at most ``review_budget`` tasks across many opportunities.

        1. screen every deal (record, account, data quality, score);
        2. rank deals by investigation value and fully investigate the top ``investigate_budget``;
        3. order the resulting review tasks by the documented triage key and keep the budget.
        Investigation value: borderline 1.0, at risk 0.8, favourable 0.3, blocked 0 (data queue), invalid 0.
        Triage key: (escalation before uncertain review, higher evidence quality, lower P(Won), longer open, ID).
        """
        goal={'id':'PRIORITIZE_REVIEW_QUEUE','description':f'Choose which of {len(cases)} opportunities need human review first (budget {review_budget})'}
        screen_tools={'get_opportunity','get_account_information','check_data_quality','predict_win_probability'}&self.config.allowed()
        screened=[self.run(c,goal=goal,allowed=screen_tools,stop_after_screen=True) for c in cases]
        def value(ep):
            b=ep.state['beliefs']
            if ep.state['status'].get('opportunity')!='known' or planner.blocked(b): return 0.
            return {'borderline':1.,'at_risk':.8,'favorable':.3}.get(planner.risk_signal(b.get('prediction')),0.)
        def days(ep): return (ep.state['beliefs'].get('opportunity') or {}).get('days_open_at_snapshot',0)
        order=sorted(screened,key=lambda ep:(-value(ep),-days(ep),ep.case.opportunity_id))
        budget=len(order) if investigate_budget is None else investigate_budget
        traces=[]; deferred=[]
        for rank,ep in enumerate(order):
            blocked_or_invalid=value(ep)==0.
            if rank<budget or blocked_or_invalid:
                traces.append(self.run(ep.case,prior=ep,goal=goal,write=write))
            else:
                deferred.append({'opportunity_id':ep.case.opportunity_id,'investigation_value':value(ep),
                                 'reason':'investigation budget exhausted; screened only, no task created'})
        def key(t):
            a=t['decision'].get('assessment') or {}; p=t['decision'].get('prediction') or {}
            tier={'ESCALATE_AT_RISK_REVIEW':0,'REVIEW_UNCERTAIN':1}.get(t['decision']['action'],9)
            return (tier,-(a.get('evidence_quality') or 0),p.get('win_probability',1),
                    -next((s['tool_call']['result'].get('days_open_at_snapshot',0) for s in t['steps']
                           if (s.get('tool_call') or {}).get('tool')=='get_opportunity' and s['tool_call']['status']=='ok'),0),
                    t['case']['opportunity_id'])
        sales=sorted([t for t in traces if t['decision']['action'] in ('ESCALATE_AT_RISK_REVIEW','REVIEW_UNCERTAIN')],key=key)
        queue=[{'rank':i+1,'opportunity_id':t['case']['opportunity_id'],'decision':t['decision']['action'],
                'win_probability':(t['decision'].get('prediction') or {}).get('win_probability'),
                'evidence_quality':(t['decision'].get('assessment') or {}).get('evidence_quality'),
                'within_budget':i<review_budget} for i,t in enumerate(sales)]
        summary={'goal':goal,'policy_version':policy.POLICY_VERSION,'configuration':self.config.name,
                 'cases':len(cases),'investigated':len(traces),'deferred':deferred,'review_budget':review_budget,
                 'sales_review_queue':queue,
                 'data_completion_queue':[t['case']['opportunity_id'] for t in traces if t['decision']['action']=='DATA_COMPLETION'],
                 'monitor_only':[t['case']['opportunity_id'] for t in traces if t['decision']['action']=='MONITOR_NO_TASK'],
                 'abstained':[t['case']['opportunity_id'] for t in traces if t['decision']['action']=='ABSTAIN'],
                 'decision_counts':pd.Series([t['decision']['action'] for t in traces]).value_counts().to_dict() if traces else {},
                 'triage_key':'escalation before uncertain review; higher evidence quality; lower P(Won); longer open; ID',
                 'note':'Probabilities only break ties inside an evidence tier; they are not treated as business utility.'}
        summary=jsonable(summary); summary['content_sha256']=content_hash({**summary,'traces':[t['content_sha256'] for t in traces]})
        if self.config.tool_execution and write:
            path=self.guard.local_path(self.output_dir/'batch_summary.json'); path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
        return summary,traces
