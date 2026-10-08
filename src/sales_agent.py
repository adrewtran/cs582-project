"""Deterministic, inspectable decision support. No autonomous action or LLM API."""
import json
import math
from pathlib import Path
import pandas as pd
from src.history import HISTORY_COLUMNS


def recommend(opportunity,probability,history,factors,warnings):
    p=float(probability)
    if not math.isfinite(p) or not 0<=p<=1: raise ValueError('probability must be finite in [0,1]')
    risk='HIGH' if p<.4 else ('MEDIUM' if p<.7 else 'LOW')
    actions=[]; notices=list(warnings)
    def add(rule,title,evidence,rationale):
        actions.append(dict(rule_id=rule,action=title,evidence=evidence,rationale=rationale))
    account=opportunity.get('account')
    if account is None or pd.isna(account) or str(account).strip() in ('','Missing'):
        notices.append('Missing account: outside the complete-account training support.')
        add('complete_account','Confirm the account identity and complete the CRM record.',
            {'account_missing':True},'Customer context is unavailable; a score is not a substitute for missing data.')
    elif history.get('account_closed_count',0)<5:
        add('thin_account_history','Verify customer needs directly before using the score.',
            {'prior_closed_count':float(history.get('account_closed_count',0))},
            'Fewer than five eligible prior account outcomes provide weak account-specific evidence.')
    global_rate=float(history.get('global_win_rate',.5))
    if history.get('product_closed_count',0)>=5 and history.get('product_win_rate',1)<global_rate-.1:
        add('product_review','Review product fit and objections with the account owner.',
            {'prior_closed_count':float(history['product_closed_count']),
             'smoothed_win_rate':float(history['product_win_rate']),'eligible_global_win_rate':global_rate},
            'Past product outcomes lag the eligible archive by more than 10 percentage points; this is association, not a cause.')
    if risk=='HIGH' and history.get('agent_closed_count',0)>=5 and history.get('agent_win_rate',1)<global_rate-.1:
        add('senior_review','Request a second review of qualification and sales approach.',
            {'prior_closed_count':float(history['agent_closed_count']),
             'smoothed_agent_win_rate':float(history['agent_win_rate']),'eligible_global_win_rate':global_rate,'p_win':p},
            'Both the model score and prior context warrant review; do not use this as an employee performance judgment.')
    reasons=[{'feature':str(k),'reference_delta':float(v),'direction':'supports win' if v>0 else 'opposes win'}
             for k,v in sorted(factors.items(),key=lambda item:(-abs(item[1]),item[0]))
             if math.isfinite(float(v)) and abs(v)>1e-10][:3]
    if len(actions)<2 and any(r['feature'] in ('product','sales_price') for r in reasons):
        add('product_price_context','Confirm the recorded product and list-price context.',
            {'reference_sensitivities':[r for r in reasons if r['feature'] in ('product','sales_price')]},
            'The score is sensitive to this input. It does not show that changing price would improve the outcome.')
    # Two optional evidence-triggered rules + two universal guardrails = 2..4 actions.
    actions=actions[:2]
    add('verify_status','Check that this historical CRM opportunity is still active.',
        {'record_stage':opportunity.get('deal_stage','not supplied')},
        'This dataset is a static snapshot; current contact activity and buyer intent are not observed.')
    add('human_review','Review the score, evidence quality and reasons before taking action.',
        {'p_win':p,'loss_risk':risk},'Risk bands are fixed heuristics, not validated revenue or intervention policies.')
    if history.get('global_closed_count',0)==0: notices.append('No eligible past closures; history uses explicit cold-start defaults.')
    return {'model_prediction':{'win_probability':p,'loss_probability':1-p,'key_factors':reasons,
                'explanation_method':'single-feature reference sensitivity; nonadditive and noncausal'},
            'agent_recommendation':{'loss_risk':risk,'actions':actions,'warnings':list(dict.fromkeys(notices)),
                'history_context':{k:float(v) for k,v in history.items() if k in HISTORY_COLUMNS},
                'policy':'deterministic rules v1; fixed risk cutoffs 0.40/0.70; support minimum 5',
                'causal_claim':False,'autonomous_execution':False}}


def timing_context(engage_date,available):
    retrospective=pd.Timestamp(engage_date)<pd.Timestamp(available)
    return {'engagement_date':str(pd.Timestamp(engage_date).date()),'model_available_date':available,
            'scoring_context':'retrospective_snapshot' if retrospective else 'post_model_engagement_snapshot',
            'warning':'Model did not exist at original engagement.' if retrospective else 'Static historical data; not a live deployment.'}


def select_demo_row(rows,available):
    eligible=rows.loc[rows.engage_date>=pd.Timestamp(available)]
    if len(eligible):
        complete=eligible.loc[eligible.account.notna()]
        eligible=complete if len(complete) else eligible
    else: eligible=rows
    if eligible.empty: raise ValueError('No dated Engaging opportunity is available')
    return eligible.sort_values(['engage_date','opportunity_id'],kind='stable').iloc[0]


def export_agent_outputs(out,history_dataset):
    out=Path(out); manifest=json.loads((out/'run_manifest.json').read_text())
    scores=pd.read_csv(out/'open_deal_predictions.csv')
    rows=history_dataset.extra['scorable_open_deals'].set_index('opportunity_id',drop=False)
    if scores.empty:
        (out/'sales_assistant_outputs.jsonl').write_text('',encoding='utf-8')
        (out/'demo_example.json').write_text(json.dumps({'available':False,
            'reason':'No dated Engaging opportunity is available.'},indent=2),encoding='utf-8')
        return
    exported=[]
    for score in scores.to_dict('records'):
        row=rows.loc[score['opportunity_id']]
        timing=timing_context(row.engage_date,manifest['model_available_date'])
        result=recommend(row.to_dict(),score['win_probability'],row[HISTORY_COLUMNS].to_dict(),
                         json.loads(score['reference_deltas_json']),[timing['warning']])
        result.update(opportunity={k:(None if pd.isna(row[k]) else str(row[k]))
            for k in ['opportunity_id','account','product','sales_agent','deal_stage']},timing=timing,
            model={'experiment':manifest['selected_experiment'],'name':manifest['selected_model']})
        exported.append(result)
    (out/'sales_assistant_outputs.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in exported)+'\n',encoding='utf-8')
    chosen=select_demo_row(rows.reset_index(drop=True),manifest['model_available_date']).opportunity_id
    example=next(r for r in exported if r['opportunity']['opportunity_id']==chosen)
    (out/'demo_example.json').write_text(json.dumps(example,indent=2,ensure_ascii=False),encoding='utf-8')
