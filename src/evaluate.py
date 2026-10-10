"""Evaluate stage: test metrics, figures, explanations and audits for already-trained models.

    python -m src.evaluate [--output PATH]

Loads models saved by src.train. Nothing here can change the selected model, threshold or calibration.
"""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from src.data.crm import build
from src.data.split import asof_split
from src.evaluation import audits,figures,novelty_checks
from src.evaluation.metrics import metrics
from src.explain import importance
from src.models import bundle
from src.outputs import DEFAULT_OUTPUT,Outputs

LIMITATIONS=[
    'Previously inspected dataset: exploratory holdout, not an untouched external test.',
    'Outcome-availability purge reduces but cannot remove right-censoring/closed-only selection bias.',
    'Static account/product/team snapshots may not reflect historical values at engagement.',
    'Repeated accounts/agents cross periods: not a new-customer generalization test.',
    'Validation reused for early stopping, selection, threshold and calibration; estimates may be optimistic.',
    'Open scoring is a frozen-model snapshot demo, not an as-of replay of each historical open deal.',
    'Missing-account open deals are out of training support; no validated uplift or causal recommendation.',
    'Probability calibration and priority thresholds need external/prospective validation.']


def run(output_dir=DEFAULT_OUTPUT,trained=None,dataset=None,split=None):
    out=Outputs(output_dir).make(); manifest=out.read_manifest()
    if 'selected_model' not in manifest: raise RuntimeError(f'{out.root} has no trained models: run src.train first')
    quick=manifest['mode']!='full'
    trained=trained or bundle.load_trained(out)
    dataset=dataset or build(); split=split or asof_split(dataset)
    X=dataset.features(); y=dataset.labels()
    xt,yt=X.loc[split.test],y.loc[split.test]; xv,yv=X.loc[split.validation],y.loc[split.validation]
    selected=trained.selected; calibrated=trained.calibrated
    figures.eda(dataset,split,out)
    probabilities={name:model.predict_proba(xt)[:,1] for name,model in trained.models.items()}
    rows=[{'model':name,**metrics(yt,p,trained.thresholds[name])} for name,p in probabilities.items()]
    pd.DataFrame(rows).to_csv(out.metrics/'test_metrics.csv',index=False)
    pc=calibrated.predict_proba(xt)[:,1]
    pd.DataFrame([{'variant':'raw',**metrics(yt,probabilities[selected],trained.thresholds[selected])},
                  {'variant':'calibrated',**metrics(yt,pc,trained.decision_threshold)}]).to_csv(out.metrics/'calibration_test.csv',index=False)
    test_predictions=dataset.frame.loc[split.test,['opportunity_id','deal_stage','engage_date']].copy()
    for name,p in probabilities.items(): test_predictions['p_'+name]=p
    test_predictions['p_selected_calibrated']=pc; test_predictions.to_csv(out.metrics/'test_predictions.csv',index=False)
    histories={name:{'settings':m.settings,'history':m.history,'fit_seconds':m.fit_seconds} for name,m in trained.models.items()}
    figures.learning_curves(histories,out)
    figures.evaluation_figures(yt,probabilities,trained.thresholds,{'raw':probabilities[selected],'calibrated':pc},out)
    importance.native_importances(trained.models,out)
    importance.permutation_importance(calibrated,xt,yt,out,quick=quick)
    importance.shap_audit(trained.models['random_forest'],xt,test_predictions.opportunity_id,out,quick=quick)
    audits.leakage_control(dataset,split,next(r for r in rows if r['model']==selected),out)
    audits.priority_checks(yt,pc,yv,calibrated.predict_proba(xv)[:,1],out)
    print('Checking split protocols and explanations...',flush=True)
    protocols=novelty_checks.split_protocols(dataset,split,out,quick=quick)
    checks=novelty_checks.explanation_checks(trained.models['random_forest'],calibrated,dataset,split,trained.reference,out,quick=quick)
    manifest['novelty_checks']={'split_protocol_max_auc_gain':float(protocols.delta_vs_asof.max()),
        'rf_explanation_spearman_median':checks['agreement']['spearman_median'],
        'deletion_ratio_top_to_random':checks['deletion']['ratio_top_to_random']}
    manifest['limitations']=LIMITATIONS
    manifest.setdefault('stages',{})['evaluate']=datetime.now(timezone.utc).isoformat()
    out.write_manifest(manifest)
    print(f'EVALUATED: {selected} test ROC-AUC {next(r for r in rows if r["model"]==selected)["roc_auc"]:.4f}',flush=True)
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=None,help='Run folder written by src.train')
    run(parser.parse_args().output or DEFAULT_OUTPUT)


if __name__=='__main__': main()
