"""Planning: which tool to run next, and which final action the gathered evidence can support.

Tool choice (one step at a time, one-step lookahead):
    utility(tool) = need(slot it fills | current beliefs) - cost(tool)
    run the applicable tool with the highest utility if utility >= MIN_GAIN; ties follow INFORMATION_TOOLS order;
    when no tool clears MIN_GAIN, assess the evidence once, then decide.

Needs of the observation tools are fixed: opportunity 1.00, account 0.80, data quality 0.90, prediction 0.90
(0.30 when a blocking data issue exists: kept only as labelled context).

Needs of the evidence tools (history, model agreement, explanation) come from value of information. The planner
re-runs the decision rule on hypothetical beliefs where the missing slot takes its best or its worst plausible
value. If the chosen final action differs, the slot is decision-relevant (history 0.80, agreement and explanation
0.70). Otherwise, the slot still has 0.50 when the decision will create a sales review task (the reviewer needs the
context), and 0.05 when it cannot matter (e.g. a blocked or invalid record).

Final action: the first of policy.DECISIONS whose requirements all hold, i.e. the most specific action that the
evidence supports. Every requirement is listed per candidate in the trace.
"""
from copy import deepcopy
from src.agent import policy
from src.agent.tools import INFORMATION_TOOLS,TOOLS

SLOT={name:TOOLS[name].provides for name in INFORMATION_TOOLS}
RELEVANT={'history':.80,'agreement':.70,'explanation':.70}
TASK_CONTEXT=.50
IRRELEVANT=.05


def risk_signal(prediction,th=policy.DEFAULT_THRESHOLDS):
    if not prediction: return None
    d=prediction['distance_to_threshold']
    return 'borderline' if abs(d)<th.borderline_margin else ('at_risk' if d<0 else 'favorable')


def split(agreement,th):
    """Committee split under the policy threshold (the tool reports the share)."""
    return agreement['disagreement_share']>=th.committee_split


def blocked(beliefs):
    quality=beliefs.get('data_quality')
    return bool(quality and quality['blocking'])


def reliability(card):
    low,_=card['validation_calibrated_roc_auc_ci95']; auc=card['validation_roc_auc']
    return 'unusable' if low<=.5 else 'weak' if auc<.65 else 'moderate' if auc<.75 else 'strong'


def assess(beliefs,card,th=policy.DEFAULT_THRESHOLDS):
    """Pure evidence assessment (used by the evaluate_evidence tool and by lookahead)."""
    prediction=beliefs.get('prediction'); quality=beliefs.get('data_quality') or {}
    history=beliefs.get('history'); agreement=beliefs.get('agreement')
    checks={
        'account_present':beliefs.get('account') is not None,
        'account_history_supported':bool(history) and history['account_closed_count']>=th.support_min,
        'product_history_supported':bool(history) and history['product_closed_count']>=th.support_min,
        'agent_history_supported':bool(history) and history['agent_closed_count']>=th.support_min,
        'no_imputed_inputs':bool(quality) and not quality.get('imputed_inputs'),
        'inputs_in_training_support':bool(quality) and not {'unseen_category','outside_training_range'}&set(quality.get('warnings',[])),
        'committee_agrees':bool(agreement) and not split(agreement,th),
    }
    score=round(sum(policy.EVIDENCE_WEIGHTS[k] for k,v in checks.items() if v),10)
    risk=risk_signal(prediction,th); conflicts=[]
    if prediction and history and history['account_closed_count']>=th.support_min:
        gap=round(history['account_win_rate']-history['global_win_rate'],10)
        if (risk=='favorable' and gap<=-th.conflict_gap) or (risk=='at_risk' and gap>=th.conflict_gap):
            conflicts.append({'code':f'account_history_contradicts_{risk}_score','account_win_rate':history['account_win_rate'],
                              'archive_win_rate':history['global_win_rate'],'gap':gap})
    if prediction and agreement and split(agreement,th):
        conflicts.append({'code':'model_committee_split','disagreeing_models':agreement['disagreeing_models'],
                          'disagreement_share':agreement['disagreement_share']})
    return {'evidence_quality':score,'evidence_checks':checks,'sufficient':score>=th.sufficient_evidence,
            'blocking_issues':list(quality.get('blocking',[])),'conflicts':conflicts,'risk_signal':risk,
            'model_reliability':reliability(card),'validation_roc_auc':card['validation_roc_auc'],
            'validation_roc_auc_ci95':card['validation_calibrated_roc_auc_ci95'],
            'rules':f'evidence weights {policy.EVIDENCE_WEIGHTS}; sufficient >= {th.sufficient_evidence}; '
                    f'borderline margin {th.borderline_margin}; reliability from the validation AUC CI only'}


def decision_candidates(state,reliability_checks=True,card=None,th=policy.DEFAULT_THRESHOLDS):
    """Every final action with its requirements evaluated against the current beliefs."""
    beliefs=state['beliefs']
    a=beliefs.get('assessment') or (assess(beliefs,card,th) if card and beliefs.get('prediction') else {})
    has_prediction=state['status'].get('prediction')=='known'
    risk=a.get('risk_signal') or risk_signal(beliefs.get('prediction'),th)
    gate=lambda value:value if reliability_checks else True   # ablation: reliability evidence is ignored
    common={'prediction_available':has_prediction,'no_blocking_data_issue':gate(not a.get('blocking_issues')),
            'evidence_sufficient':gate(bool(a.get('sufficient'))),'no_conflicting_evidence':gate(not a.get('conflicts')),
            'model_not_indistinguishable_from_chance':gate(a.get('model_reliability')!='unusable')}
    rows=[
        ('ESCALATE_AT_RISK_REVIEW',{**common,'score_at_risk':risk=='at_risk',
                                     'reasons_available':state['status'].get('explanation')=='known'}),
        ('MONITOR_NO_TASK',{**common,'score_favorable':risk=='favorable'}),
        ('REVIEW_UNCERTAIN',{'prediction_available':has_prediction,'no_blocking_data_issue':common['no_blocking_data_issue']}),
        ('DATA_COMPLETION',{'blocking_data_issue':bool(a.get('blocking_issues')) if reliability_checks else False}),
        ('ABSTAIN',{}),
    ]
    return [dict(decision=d,requirements=req,valid=all(req.values())) for d,req in rows]


def choose_decision(candidates):
    return next(c for c in candidates if c['valid'])


def hypotheses(slot,beliefs,th=policy.DEFAULT_THRESHOLDS):
    """Best and worst plausible results of a slot, in the same schema as the real tool output."""
    if slot=='history':
        g=.6   # any archive rate: support and conflict depend only on counts and the gap
        def history(count,gap):
            h={f'{e}_closed_count':float(count) for e in ('agent','account','product')}
            h.update({f'{e}_win_rate':g for e in ('agent','account','product')},global_win_rate=g)
            h['account_win_rate']=g+gap; return h
        return [history(th.support_min,0.),history(0,0.),history(th.support_min,-th.conflict_gap),
                history(th.support_min,th.conflict_gap)]
    if slot=='agreement':
        return [{'committee_split':split,'disagreeing_models':[],'disagreement_share':float(split)} for split in (False,True)]
    if slot=='explanation': return [{'factors':[]},None]
    raise ValueError(slot)


def decision_with(state,slot,value,config,card):
    trial={'beliefs':deepcopy(state['beliefs']),'status':dict(state['status'])}
    trial['beliefs'].pop('assessment',None)
    if value is None: trial['status'][slot]='failed'
    else: trial['beliefs'][slot]=value; trial['status'][slot]='known'
    return choose_decision(decision_candidates(trial,config['reliability_checks'],card,config['thresholds']))['decision']


def value_of_information(slot,state,config,card):
    current=decision_with(state,slot,None,config,card)
    outcomes=sorted({decision_with(state,slot,v,config,card) for v in hypotheses(slot,state['beliefs'],config['thresholds'])}|{current})
    if len(outcomes)>1:
        return RELEVANT[slot],f'decision-relevant: the result could change the action ({" vs ".join(outcomes)})'
    if current in ('ESCALATE_AT_RISK_REVIEW','REVIEW_UNCERTAIN'):
        return TASK_CONTEXT,f'cannot change the action ({current}), but the review task needs this context'
    return IRRELEVANT,f'cannot change the action ({current}) and no sales review task needs it'


def need(slot,state,config,card):
    """Return (need, reason) or (None, reason) when the tool's preconditions do not hold."""
    beliefs,status=state['beliefs'],state['status']; allowed=config['allowed']
    opportunity=beliefs.get('opportunity')
    if slot=='opportunity': return 1.0,'the goal needs the opportunity record'
    if status.get('opportunity')!='known': return None,'needs the opportunity record first'
    if slot=='account':
        if opportunity['account'] is None: return None,'not applicable: the opportunity has no account key'
        return .80,'customer context and a check that the account exists'
    if slot=='data_quality':
        if opportunity['account'] is not None and status.get('account') not in ('known','failed') and 'get_account_information' in allowed:
            return None,'waits for the account lookup'
        return .90,'a score must not be trusted before its inputs are checked'
    if slot=='prediction':
        if status.get('data_quality') not in ('known','failed') and 'check_data_quality' in allowed:
            return None,'waits for the data-quality check'
        if blocked(beliefs): return .30,'context only: a blocking data issue puts the score out of training support'
        return .90,'the goal needs the model estimate'
    if status.get('prediction') not in ('known','failed'): return None,'needs a prediction attempt first'
    if slot in ('agreement','explanation') and status.get('prediction')!='known': return None,'needs a successful prediction'
    return value_of_information(slot,state,config,card)


def tool_candidates(state,allowed,card,reliability_checks=True,fixed_sequence=False,th=policy.DEFAULT_THRESHOLDS):
    rows=[]; config={'allowed':allowed,'reliability_checks':reliability_checks,'thresholds':th}
    for name in INFORMATION_TOOLS:
        slot=SLOT[name]; cost=policy.TOOL_COST[name]
        if state['status'].get(slot) in ('known','failed'): continue
        if name not in allowed:
            rows.append(dict(tool=name,slot=slot,need=None,cost=cost,utility=None,applicable=False,reason='disabled in this configuration'))
            continue
        value,reason=need(slot,state,config,card)
        if value is None:
            rows.append(dict(tool=name,slot=slot,need=None,cost=cost,utility=None,applicable=False,reason=reason)); continue
        utility=round(value-cost,10)
        if fixed_sequence: utility,reason=1.0,'fixed sequence: every available tool runs (planning ablated)'
        rows.append(dict(tool=name,slot=slot,need=value,cost=cost,utility=utility,applicable=True,reason=reason))
    return rows


def choose_tool(candidates,th=policy.DEFAULT_THRESHOLDS):
    best=[c for c in candidates if c['applicable'] and c['utility']>=th.min_gain]
    if not best: return None
    top=max(c['utility'] for c in best)
    return next(c for c in best if c['utility']==top)
