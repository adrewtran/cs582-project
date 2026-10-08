"""Real-model CRM demo. Only load a bundle created by this trusted project."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from src.datasets.crm import build
from src.history import build_history,HISTORY_COLUMNS
from src.explain import reference_sensitivities
from src.sales_agent import recommend,timing_context,select_demo_row


def demo(bundle_path,opportunity_id=None):
    bundle=joblib.load(bundle_path)
    rows=build().extra['scorable_open_deals']
    if opportunity_id is None:
        row=select_demo_row(rows,bundle['model_available_date'])
    else:
        selected=rows.loc[rows.opportunity_id.eq(opportunity_id)]
        if len(selected)!=1: raise ValueError('ID must identify one dated Engaging opportunity (not a closed or unknown deal)')
        row=selected.iloc[0]
    query=rows.loc[[row.name]]
    history,audit=build_history(query,bundle['history_archive'])
    inputs=query.join(history)[bundle['features']]
    p=float(bundle['model'].predict_proba(inputs)[0,1])
    factors=reference_sensitivities(bundle['model'],inputs,bundle['reference']).iloc[0].to_dict()
    timing=timing_context(row.engage_date,bundle['model_available_date'])
    result=recommend(row.to_dict(),p,history.iloc[0].to_dict(),factors,[timing['warning']])
    result.update(opportunity={k:None if pd.isna(row[k]) else str(row[k])
        for k in ['opportunity_id','account','product','sales_agent','deal_stage']},timing=timing,
        model={'experiment':bundle['experiment'],'name':bundle['model'].base.name,
               'history_used_by_predictor':bundle['experiment']=='history'},
        history_audit={'eligible_archive_count':int(audit.iloc[0].eligible_archive_count),
                       'latest_close_date':None if pd.isna(audit.iloc[0].latest_close_date) else str(audit.iloc[0].latest_close_date.date()),
                       'archive_source':'frozen_train_only'})
    result['model_prediction']['predicted_outcome']='Won' if p>=bundle['decision_threshold'] else 'Lost'
    result['model_prediction']['decision_threshold']=bundle['decision_threshold']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,default=Path('reports/crm/final/model_bundle.joblib'))
    parser.add_argument('--opportunity-id')
    parser.add_argument('--json',action='store_true')
    args=parser.parse_args(); result=demo(args.bundle,args.opportunity_id)
    if args.json: print(json.dumps(result,indent=2,ensure_ascii=False)); return
    prediction=result['model_prediction']; agent=result['agent_recommendation']
    print('CRM SALES ASSISTANT — REAL SAVED MODEL')
    print(json.dumps(result['opportunity'],indent=2))
    print(f"Model: {result['model']['experiment']}/{result['model']['name']}")
    print(f"Win: {prediction['win_probability']:.1%} | Loss: {prediction['loss_probability']:.1%} | Loss risk: {agent['loss_risk']}")
    print(f"Timing: {result['timing']['scoring_context']} | {result['timing']['warning']}")
    print('Reasons (reference sensitivities, not causes):')
    for reason in prediction['key_factors']: print(f"  {reason['feature']}: {reason['reference_delta']:+.4f}")
    print('Evidence-backed review actions (not model outputs):')
    for action in agent['actions']:
        print(f"  [{action['rule_id']}] {action['action']}\n    Evidence: {json.dumps(action['evidence'])}\n    Why: {action['rationale']}")
    for warning in agent['warnings']: print('WARNING:',warning)


if __name__=='__main__': main()
