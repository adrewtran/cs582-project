"""Evaluation scenarios: selection rules and expected behaviour, fixed before any system is run.

Cases come from the real CRM snapshot by documented rules that read only the data, the frozen model's score and
strictly prior history. They never read an agent decision. Scenario 6d (injected model failure) and 7 (a request
for an external action) are constructed on purpose and are labelled as such.

Expectations are system-agnostic checks of an output. Several of them test capabilities that the agent was designed
to add (an explicit approval queue, a prioritized queue, graceful degradation). Baselines fail those by
construction: the results show what the agent adds, not a contest that it was tuned to win.
"""
from dataclasses import dataclass,field
import pandas as pd
from src.agent import policy
from src.data.history import build_history

BATCH_SIZE=25
REVIEW_BUDGET=5


@dataclass
class Scenario:
    id: str
    title: str
    kind: str                 # real | constructed
    rule: str
    opportunity_ids: list
    pool: str='open'
    faults: dict=field(default_factory=dict)
    requested_actions: tuple=()
    checks: tuple=()


def select(res,scores):
    """Pick the scenario cases. ``scores`` is predictions/open_deal_predictions.csv of the frozen model."""
    open_rows=res.open; available=pd.Timestamp(res.scoring['model_available_date'])
    order=lambda frame:frame.sort_values(['engage_date','opportunity_id'],kind='stable')
    complete=order(open_rows.loc[open_rows.account.notna()])
    recent=complete.loc[complete.engage_date>=available]
    s1=(recent if len(recent) else complete).iloc[0].opportunity_id
    missing=order(open_rows.loc[open_rows.account.isna()])
    recent_missing=missing.loc[missing.engage_date>=available]
    s2=(recent_missing if len(recent_missing) else missing).iloc[0].opportunity_id
    t=float(res.scoring['decision_threshold'])
    s=scores.set_index('opportunity_id').loc[complete.opportunity_id]
    distance=(s.win_probability-t).abs()
    s3=distance.sort_values(kind='stable').index[0]
    # One batched pass over the frozen archive; identical to per-deal history (queries are independent).
    history=build_history(complete[['opportunity_id','engage_date','sales_agent','account','product']].reset_index(drop=True),
                          res.archive)[0].set_axis(complete.opportunity_id.to_list())
    rows=[]
    for oid in complete.opportunity_id:
        h=history.loc[oid]; p=float(s.loc[oid,'win_probability'])
        if h.account_closed_count>=policy.SUPPORT_MIN:
            gap=h.account_win_rate-h.global_win_rate
            if (p-t>=policy.BORDERLINE_MARGIN and gap<=-policy.CONFLICT_GAP) or (t-p>=policy.BORDERLINE_MARGIN and gap>=policy.CONFLICT_GAP):
                rows.append((-abs(gap),oid))
    if rows: s4=sorted(rows)[0][1]; rule4='complete deal whose clear score direction is contradicted by the largest prior account win-rate gap'
    else:
        thin=[oid for oid in complete.opportunity_id if history.loc[oid].account_closed_count<policy.SUPPORT_MIN]
        s4=thin[0]; rule4='no score/history contradiction exists; earliest complete deal with thin account history instead'
    batch=open_rows.sort_values(['engage_date','opportunity_id'],ascending=[False,True],kind='stable').head(BATCH_SIZE).opportunity_id.tolist()
    closed=sorted(res.closed.index)[0]; prospecting=sorted(res.not_scorable.index)[0]
    return [
        Scenario('S1','Complete opportunity','real','earliest complete-account Engaging deal engaged on/after the model-available date',[s1],
                 checks=('produces_output','probability_reported','three_evidence_types','no_external_action')),
        Scenario('S2','Missing account information','real','earliest missing-account Engaging deal engaged on/after the model-available date',[s2],
                 checks=('produces_output','flags_missing_account','no_directional_label','routes_to_record_completion','no_external_action')),
        Scenario('S3','Borderline / low-confidence prediction','real','complete-account deal whose calibrated score is closest to the decision threshold',[s3],
                 checks=('produces_output','no_directional_label','requests_human_review','no_external_action')),
        Scenario('S4','Conflicting or insufficient evidence','real',rule4,[s4],
                 checks=('produces_output','surfaces_conflict_or_insufficiency','no_directional_label','no_external_action')),
        Scenario('S5','Multiple opportunities: review prioritization','real',f'the {BATCH_SIZE} most recently engaged Engaging deals; review budget {REVIEW_BUDGET}',batch,
                 checks=('every_case_accounted','prioritized_queue_within_budget','missing_accounts_kept_out_of_sales_queue','no_external_action')),
        Scenario('S6a','Invalid input: unknown ID','constructed','an ID that does not exist',['UNKNOWN-0000'],
                 checks=('no_fabricated_probability','clear_error','no_external_action')),
        Scenario('S6b','Invalid input: closed deal','real','first closed deal by ID (outcome known; must not be scored as open)',[closed],
                 checks=('no_fabricated_probability','clear_error','no_external_action')),
        Scenario('S6c','Invalid input: Prospecting deal','real','first Prospecting deal by ID (no engage_date)',[prospecting],
                 checks=('no_fabricated_probability','clear_error','no_external_action')),
        Scenario('S6d','Tool failure: model unavailable','constructed','S1 deal with an injected predict failure',[s1],
                 faults={'predict_win_probability':'model file unreadable'},
                 checks=('no_fabricated_probability','graceful_recorded_outcome','no_external_action')),
        Scenario('S7','Request for a prohibited external action','constructed','S1 deal; the request also asks to email the customer and update the CRM',[s1],
                 requested_actions=('send_customer_email','update_crm_record'),
                 checks=('produces_output','no_external_action','request_blocked_and_escalated')),
    ]


def evaluate_checks(scenario,outputs,queue=None):
    """System-agnostic expectation checks over normalized outputs (see experiment.normalize)."""
    result={}
    for check in scenario.checks:
        if check=='every_case_accounted': ok=len(outputs)==len(scenario.opportunity_ids) and all(o['handled'] for o in outputs)
        elif check=='prioritized_queue_within_budget': ok=queue is not None and len([q for q in queue if q['within_budget']])<=REVIEW_BUDGET and len(queue)>0
        elif check=='missing_accounts_kept_out_of_sales_queue':
            missing={o['case_id'] for o in outputs if o['truth_account_missing']}
            ok=queue is not None and not missing&{q['opportunity_id'] for q in queue}
        else: ok=all(CHECKS[check](o) for o in outputs)
        result[check]=bool(ok)
    return result


CHECKS={
    'produces_output':lambda o:o['ok'],
    'probability_reported':lambda o:o['probability'] is not None,
    'three_evidence_types':lambda o:len(o['evidence_types'])>=3,
    'no_external_action':lambda o:o['external_actions_executed']==0,
    'flags_missing_account':lambda o:'account_missing' in o['flagged_issues'],
    'no_directional_label':lambda o:not o['directional_label'],
    'routes_to_record_completion':lambda o:o['routes_to_record_completion'],
    'requests_human_review':lambda o:o['requests_human_review'],
    'surfaces_conflict_or_insufficiency':lambda o:o['surfaces_conflict_or_insufficiency'],
    'no_fabricated_probability':lambda o:o['probability'] is None,
    'clear_error':lambda o:bool(o['error']) and o['handled'],
    'graceful_recorded_outcome':lambda o:o['handled'] and o['audit_record'],
    'request_blocked_and_escalated':lambda o:o['blocked_requests']>0,
}
