"""One command: python -m src.run_project [--quick] [--output PATH]."""
import argparse
from datetime import datetime,timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import subprocess
import time
import joblib
import numpy as np
import pandas as pd
from src.datasets.crm import build,CRM_DIR
from src.temporal import asof_split
from src.models import MODEL_NAMES,fit_model
from src.evaluation import metrics,choose_threshold,select_model,CalibratedModel
from src.explain import make_reference,explain_open
from src import analysis

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'reports/crm/final'


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')


def prepare(dataset,split,quick=False):
    """Fit/choose using training + validation only; no test predictions or metrics."""
    xv=dataset.features().loc[split.validation]; yv=dataset.labels().loc[split.validation]
    models={}; rows=[]; thresholds={}; histories={}
    for name in MODEL_NAMES:
        print(f'Training {name} on CPU ({len(dataset.feature_columns)} features)...',flush=True)
        model=fit_model(name,dataset,split.train,split.validation,quick=quick); models[name]=model
        pv=model.predict_proba(xv)[:,1]; thresholds[name]=choose_threshold(yv,pv)
        rows.append({'model':name,**metrics(yv,pv,thresholds[name])})
        histories[name]={'settings':model.settings,'history':model.history,'fit_seconds':model.fit_seconds}
    selected=select_model(rows)
    calibrated=CalibratedModel.fit(models[selected],xv,yv)
    return dict(models=models,val_rows=rows,thresholds=thresholds,histories=histories,
                selected=selected,calibrated=calibrated)


def run(output_dir=DEFAULT_OUTPUT,quick=False,documents=True,*,dataset=None,split=None,prepared=None):
    start=time.perf_counter(); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    (out/'figures').mkdir(exist_ok=True)
    manifest={'status':'running','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full',
              'started_utc':datetime.now(timezone.utc).isoformat(),'seed':42,'python':platform.python_version(),
              'platform':platform.platform(),'machine':platform.machine(),
              'run_location':'execution environment; Google Colab has not been independently verified'}
    write_json(out/'run_manifest.json',manifest)
    try:
        dataset=dataset if dataset is not None else build()
        split=split if split is not None else asof_split(dataset)
        X=dataset.features(); y=dataset.labels()
        xv,yv=X.loc[split.validation],y.loc[split.validation]
        xt,yt=X.loc[split.test],y.loc[split.test]
        split.manifest.to_csv(out/'split_manifest.csv',index=False)
        write_json(out/'data_quality.json',dataset.report)
        manifest['features']=dataset.feature_columns
        manifest['raw_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(CRM_DIR.glob('*.csv'))}
        manifest['split']={'method':'engagement date groups + outcome availability purge',
            'validation_start':str(split.validation_start.date()),'test_start':str(split.test_start.date()),
            'train_rows':len(split.train),'validation_rows':len(split.validation),'test_rows':len(split.test),
            'purged_rows':int(split.manifest.role.str.startswith('purged').sum()),
            'train_win_rate':float(y.loc[split.train].mean()),'validation_win_rate':float(yv.mean()),'test_win_rate':float(yt.mean())}
        manifest['dependencies']={name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','torch','pytorch-tabnet','shap','catboost','matplotlib','joblib','python-pptx','python-docx']}
        git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
        manifest['base_commit']=git.stdout.strip() if git.returncode==0 else 'not a git checkout'
        # Hash source independently: working tree can be newer than the commit.
        manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}
        analysis.eda(dataset,split,out)
        fitted=prepared if prepared is not None else prepare(dataset,split,quick)
        models=fitted['models']; val_rows=fitted['val_rows']; thresholds=fitted['thresholds']; histories=fitted['histories']
        selected=fitted['selected']; calibrated=fitted['calibrated']
        test_rows=[]; probabilities={}
        mapped_threshold=float(calibrated.transform(thresholds[selected]))
        manifest.update(selected_model=selected,selection_rule='validation ROC-AUC, then lower Brier, then model name',
            calibration={'method':'positive-slope sigmoid on validation logit scores; base remains frozen',
                         'slope':calibrated.slope,'intercept':calibrated.intercept},
            raw_decision_threshold=thresholds[selected],calibrated_decision_threshold=mapped_threshold,
            threshold_rule='validation macro-F1; 0.10..0.90 by 0.01; ties closest to 0.50',
            priority={'medium_min':.4,'high_min':.7,'status':'unvalidated heuristic, not a business policy'})
        for name,model in models.items():
            p=model.predict_proba(xt)[:,1]; probabilities[name]=p
            test_rows.append({'model':name,**metrics(yt,p,thresholds[name])})
        pd.DataFrame(val_rows).to_csv(out/'validation_metrics.csv',index=False)
        pd.DataFrame(test_rows).to_csv(out/'test_metrics.csv',index=False)
        write_json(out/'model_training.json',histories)
        analysis.learning_curves(histories,out)
        pc=calibrated.predict_proba(xt)[:,1]
        calibration_rows=[{'variant':'raw',**metrics(yt,probabilities[selected],thresholds[selected])},
                          {'variant':'calibrated',**metrics(yt,pc,mapped_threshold)}]
        pd.DataFrame(calibration_rows).to_csv(out/'calibration_test.csv',index=False)
        test_predictions=dataset.frame.loc[split.test,['opportunity_id','deal_stage','engage_date']].copy()
        for name,p in probabilities.items(): test_predictions['p_'+name]=p
        test_predictions['p_selected_calibrated']=pc; test_predictions.to_csv(out/'test_predictions.csv',index=False)
        analysis.evaluation_figures(yt,probabilities,thresholds,{'raw':probabilities[selected],'calibrated':pc},out)
        analysis.native_importances(models,out)
        analysis.permutation_importance(calibrated,xt,yt,out,quick=quick)
        analysis.leakage_control(dataset,split,next(r for r in test_rows if r['model']==selected),out)
        analysis.shap_audit(models['random_forest'],xt,test_predictions.opportunity_id,out,quick=quick)
        analysis.priority_checks(yt,pc,yv,calibrated.predict_proba(xv)[:,1],out)
        reference=make_reference(X.loc[split.train],dataset)
        write_json(out/'explanation_reference.json',reference)
        predictions=explain_open(calibrated,dataset,reference,mapped_threshold)
        predictions.to_csv(out/'open_deal_predictions.csv',index=False)
        has_history='history_archive' in dataset.extra
        bundle={'model':calibrated,'features':dataset.feature_columns,'reference':reference,
                     'experiment':'history' if has_history else 'raw','model_available_date':str(split.test_start.date()),
                     'history_archive':dataset.extra.get('history_archive',dataset.frame.loc[split.train].copy()),
                     'decision_threshold':mapped_threshold,'warning':'Trusted local artifact only; never unpickle an untrusted file'}
        joblib.dump(bundle,out/'model_bundle.joblib',compress=3)
        manifest['experiment']=bundle['experiment']
        manifest['model_available_date']=bundle['model_available_date']
        if has_history:
            for key in ['history_audit','history_coverage','open_history_audit']:
                dataset.extra[key].to_csv(out/f'{key}.csv',index=False)
        manifest['scored_open_rows']=len(predictions)
        manifest['limitations']=[
            'Previously inspected dataset: exploratory holdout, not an untouched external test.',
            'Outcome-availability purge reduces but cannot remove right-censoring/closed-only selection bias.',
            'Static account/product/team snapshots may not reflect historical values at engagement.',
            'Repeated accounts/agents cross periods: not a new-customer generalization test.',
            'Validation reused for early stopping, selection, threshold and calibration; estimates may be optimistic.',
            'Open scoring is a frozen-model snapshot demo, not an as-of replay of each historical open deal.',
            'Missing-account open deals are out of training support; no validated uplift or causal recommendation.',
            'Probability calibration and priority thresholds need external/prospective validation.']
        manifest['elapsed_seconds']=time.perf_counter()-start
        # Documents consume this manifest; status changes only after every requested artifact succeeds.
        write_json(out/'run_manifest.json',manifest)
        if documents:
            from src.deliverables import build_deliverables
            build_deliverables(out)
        manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        write_json(out/'run_manifest.json',manifest)
        print(f'COMPLETE: {out}; selected={selected}; scored={len(predictions)}',flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        write_json(out/'run_manifest.json',manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='All six models, tiny budgets; not reportable results')
    parser.add_argument('--raw-only',action='store_true',help='Original raw-feature experiment only')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--no-documents',action='store_true',help='Run experiment only')
    args=parser.parse_args()
    output=args.output or (ROOT/'reports/crm/smoke' if args.quick else DEFAULT_OUTPUT)
    if args.raw_only:
        run(output,quick=args.quick,documents=not args.no_documents)
    else:
        from src.experiments import run_comparison
        run_comparison(output,quick=args.quick,documents=not args.no_documents)


if __name__=='__main__': main()
