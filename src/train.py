"""Train stage: fit all six models on each feature set, select on validation, calibrate, save the models.

    python -m src.train [--quick] [--output PATH] [--data-dir DIR] [--feature-sets raw,history]

Feature sets: ``raw`` (18 pre-close inputs) and ``history`` (raw + strictly prior agent/account/product
aggregates from the frozen training archive, src.data.history). On data/crm both are compared by default;
other data folders default to ``raw`` only.

--data-dir trains on another folder in the raw CRM format (e.g. data/crm_simulated); its results go to
reports/crm/simulated/ by default and may never be written into reports/crm/final/.

No test row is scored here: feature set, model, threshold and calibration are locked before evaluation
(models/selection_lock.json).
"""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from importlib.metadata import version
import platform
import subprocess
import numpy as np
import pandas as pd
from src.data.crm import build,CRM_DIR
from src.data.history import augment_dataset
from src.data.split import asof_split
from src.evaluation.metrics import metrics,choose_threshold
from src.evaluation.novelty_checks import bootstrap_auc
from src.explain.reference import make_reference
from src.models import bundle
from src.models.calibration import CalibratedModel
from src.models.zoo import MODEL_NAMES,fit_model
from src.outputs import ROOT,DEFAULT_OUTPUT,SMOKE_OUTPUT,SIMULATED_OUTPUT,Outputs,write_json

DEPENDENCIES=['numpy','pandas','scipy','scikit-learn','torch','pytorch-tabnet','shap','catboost','matplotlib','joblib']
FEATURE_SETS=('raw','history')
SELECTION_RULE='validation ROC-AUC, then lower validation Brier, then feature set, then model name'


def describe_run(manifest,dataset,split,data_dir):
    y=dataset.labels()
    manifest['features']=dataset.feature_columns
    manifest['data_dir']=str(data_dir.relative_to(ROOT)) if data_dir.is_relative_to(ROOT) else str(data_dir)
    # Hash the input tables (and the data dictionary when present), not derived files in the folder.
    inputs=[data_dir/f'{name}.csv' for name in ['sales_pipeline','accounts','products','sales_teams','data_dictionary']]
    manifest['raw_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.exists()}
    manifest['split']={'method':'engagement date groups + outcome availability purge',
        'validation_start':str(split.validation_start.date()),'test_start':str(split.test_start.date()),
        'train_rows':len(split.train),'validation_rows':len(split.validation),'test_rows':len(split.test),
        'purged_rows':int(split.manifest.role.str.startswith('purged').sum()),
        'train_win_rate':float(y.loc[split.train].mean()),'validation_win_rate':float(y.loc[split.validation].mean()),
        'test_win_rate':float(y.loc[split.test].mean())}
    manifest['dependencies']={name:version(name) for name in DEPENDENCIES}
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    manifest['base_commit']=git.stdout.strip() if git.returncode==0 else 'not a git checkout'
    # Hash source independently: working tree can be newer than the commit.
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}


def default_output(data_dir=CRM_DIR,quick=False):
    """Where a run goes when no --output is given; other data never defaults to the final folder."""
    if Path(data_dir).resolve()==CRM_DIR.resolve(): return SMOKE_OUTPUT if quick else DEFAULT_OUTPUT
    return SIMULATED_OUTPUT/('smoke' if quick else 'final')


def start(output_dir=DEFAULT_OUTPUT,quick=False,data_dir=CRM_DIR,dataset=None,split=None):
    """Step 1: create the run folder and manifest, load, validate and split the data."""
    data_dir=Path(data_dir).resolve()
    if data_dir!=CRM_DIR.resolve() and Path(output_dir).resolve()==DEFAULT_OUTPUT.resolve():
        raise ValueError(f'{DEFAULT_OUTPUT} holds only results on data/crm; choose another --output for {data_dir}')
    out=Outputs(output_dir).make()
    manifest={'status':'running','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full',
              'started_utc':datetime.now(timezone.utc).isoformat(),'seed':42,'python':platform.python_version(),
              'platform':platform.platform(),'machine':platform.machine(),
              'run_location':'local CPU execution; see platform and machine'}
    out.write_manifest(manifest)
    dataset=dataset or build(data_dir); split=split or asof_split(dataset)
    split.manifest.to_csv(out.data/'split_manifest.csv',index=False)
    write_json(out.data/'data_quality.json',dataset.report)
    describe_run(manifest,dataset,split,data_dir)
    return out,manifest,dataset,split


def default_feature_sets(data_dir=CRM_DIR):
    """Both feature sets on the real CRM data; other folders keep the 18 standard inputs unless asked."""
    return FEATURE_SETS if Path(data_dir).resolve()==CRM_DIR.resolve() else ('raw',)


def feature_variants(dataset,split,feature_sets=FEATURE_SETS):
    """Same rows, same split; ``history`` adds strictly prior aggregates from the training archive only."""
    unknown=set(feature_sets)-set(FEATURE_SETS)
    if unknown or not feature_sets: raise ValueError(f'feature sets must be chosen from {FEATURE_SETS}, got {feature_sets}')
    return {name:(dataset if name=='raw' else augment_dataset(dataset,split)) for name in FEATURE_SETS if name in feature_sets}


def fit_one(name,dataset,split,quick=False):
    """Step 2: fit one model on training rows; pick its threshold and score it on validation only."""
    X=dataset.features(); xv,yv=X.loc[split.validation],dataset.labels().loc[split.validation]
    model=fit_model(name,dataset,split.train,split.validation,quick=quick)
    pv=model.predict_proba(xv)[:,1]; threshold=choose_threshold(yv,pv)
    return model,threshold,{'model':name,**metrics(yv,pv,threshold)}


def select(candidates):
    """Lock the (feature set, model) pair on validation only; test metrics do not exist yet."""
    return min(candidates,key=lambda r:(-r['roc_auc'],r['brier'],r['feature_set'],r['model']))


def model_card(trained,dataset,split,quick=False):
    """Validation-only facts the agent may rely on: discrimination, training support and score scales."""
    X=dataset.features(); xv,yv=X.loc[split.validation],dataset.labels().loc[split.validation]
    train=X.loc[split.train]
    p=trained.calibrated.predict_proba(xv)[:,1]
    low,high=bootstrap_auc(yv,p,200 if quick else 2000)
    committee={}
    for name,model in trained.models.items():
        if name=='dummy_prior': continue
        scores=model.predict_proba(xv)[:,1]
        committee[name]=[float(v) for v in np.quantile(scores,np.linspace(0,1,101))]
    return {'model':trained.calibrated.name,'feature_set':trained.feature_set,
            'validation_rows':len(split.validation),'validation_roc_auc':float(next(r['roc_auc'] for r in trained.validation_rows if r['model']==trained.selected)),
            'validation_calibrated_roc_auc_ci95':[low,high],'validation_win_rate':float(yv.mean()),
            'decision_threshold':trained.decision_threshold,'model_available_date':trained.model_available_date,
            'training_period':[str(dataset.frame.loc[split.train,'engage_date'].min().date()),str(split.validation_start.date())],
            'training_categories':{col:sorted(map(str,train[col].dropna().unique())) for col in dataset.categorical},
            'training_numeric_range':{col:[float(train[col].min()),float(train[col].max())] for col in dataset.numeric if train[col].notna().any()},
            'committee_validation_quantiles':committee,
            'source':'validation and training rows only; no test row is used'}


def finish(out,manifest,dataset,split,fitted):
    """Step 3: select on validation, calibrate the frozen selected model, save every artifact.

    ``fitted`` maps each name in MODEL_NAMES to the (model, threshold, validation row) from fit_one, and
    ``dataset`` is the matching Dataset. To compare feature sets, pass ``{feature_set: fitted}`` and
    ``{feature_set: Dataset}`` instead; selection then runs over every (feature set, model) pair.
    """
    if list(fitted)==list(MODEL_NAMES): fitted={'raw':fitted}; datasets={'raw':dataset}
    else: datasets=dataset
    if set(fitted)!=set(datasets) or not set(fitted)<=set(FEATURE_SETS): raise ValueError('fitted and datasets must share feature sets')
    for name,models in fitted.items():
        if list(models)!=list(MODEL_NAMES): raise ValueError(f'fit all of {MODEL_NAMES} in order before selecting; {name} has {list(models)}')
    candidates=[dict(feature_set=fs,**f[2]) for fs,models in fitted.items() for f in models.values()]
    chosen=select(candidates); feature_set=chosen['feature_set']; selected=chosen['model']
    lock={'selected_feature_set':feature_set,'selected_model':selected,'selection_rule':SELECTION_RULE,
          'locked_utc':datetime.now(timezone.utc).isoformat(),'test_evaluated_at_selection':False,
          'candidates':candidates,
          'history_protocol':'frozen training archive; other deals must close strictly before engagement; own outcome excluded',
          'warning':'The test period was examined in earlier project iterations: exploratory, not pristine. No post-test tuning.'}
    write_json(out.models/'selection_lock.json',lock)
    dataset=datasets[feature_set]; chosen_fit=fitted[feature_set]
    X=dataset.features(); xv,yv=X.loc[split.validation],dataset.labels().loc[split.validation]
    models={name:f[0] for name,f in chosen_fit.items()}; thresholds={name:f[1] for name,f in chosen_fit.items()}
    rows=[f[2] for f in chosen_fit.values()]
    histories={name:{'settings':m.settings,'history':m.history,'fit_seconds':m.fit_seconds} for name,m in models.items()}
    calibrated=CalibratedModel.fit(models[selected],xv,yv)
    reference=make_reference(X.loc[split.train],dataset)
    alternatives={fs:{'models':{n:f[0] for n,f in fit.items()},'thresholds':{n:f[1] for n,f in fit.items()},
                      'validation_rows':[f[2] for f in fit.values()]} for fs,fit in fitted.items() if fs!=feature_set}
    trained=bundle.Trained(models=models,thresholds=thresholds,selected=selected,calibrated=calibrated,
        decision_threshold=float(calibrated.transform(thresholds[selected])),reference=reference,
        features=dataset.feature_columns,validation_rows=rows,feature_set=feature_set,
        history_archive=dataset.extra.get('history_archive',dataset.frame.loc[split.train].copy()),alternatives=alternatives,
        model_available_date=str(split.test_start.date()))
    pd.DataFrame(rows).to_csv(out.metrics/'validation_metrics.csv',index=False)
    write_json(out.models/'model_training.json',histories)
    write_json(out.models/'explanation_reference.json',reference)
    if 'history' in datasets:
        history=datasets['history'].extra
        for key in ['history_audit','history_coverage','open_history_audit']:
            history[key].to_csv(out.data/f'{key}.csv',index=False)
    bundle.save(out,trained)
    write_json(out.models/'model_card.json',model_card(trained,dataset,split,quick=manifest.get('mode')!='full'))
    manifest.update(selected_model=selected,feature_set=feature_set,features=dataset.feature_columns,
        feature_sets_compared=list(fitted),selection_rule=SELECTION_RULE,selection_lock='models/selection_lock.json',
        model_available_date=trained.model_available_date,
        calibration={'method':'positive-slope sigmoid on validation logit scores; base remains frozen',
                     'slope':calibrated.slope,'intercept':calibrated.intercept},
        raw_decision_threshold=thresholds[selected],calibrated_decision_threshold=trained.decision_threshold,
        threshold_rule='validation macro-F1; 0.10..0.90 by 0.01; ties closest to 0.50',
        priority={'medium_min':.4,'high_min':.7,'status':'unvalidated heuristic, not a business policy'},
        status='trained',stages={'train':datetime.now(timezone.utc).isoformat()})
    out.write_manifest(manifest)
    print(f'TRAINED: selected={feature_set}/{selected}; models saved in {out.models}',flush=True)
    return trained


def run(output_dir=DEFAULT_OUTPUT,quick=False,dataset=None,split=None,data_dir=CRM_DIR,feature_sets=None):
    """Returns (manifest, trained, dataset, split) so run_project can chain stages in memory.

    ``dataset`` in the result is the selected feature set's Dataset.
    """
    out,manifest,dataset,split=start(output_dir,quick,data_dir,dataset,split)
    variants=feature_variants(dataset,split,feature_sets or default_feature_sets(data_dir))
    fitted={}
    for feature_set,data in variants.items():
        fitted[feature_set]={}
        for name in MODEL_NAMES:
            print(f'Training {name} on CPU ({feature_set}: {len(data.feature_columns)} features)...',flush=True)
            fitted[feature_set][name]=fit_one(name,data,split,quick=quick)
    trained=finish(out,manifest,variants,split,fitted)
    return manifest,trained,variants[trained.feature_set],split


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick',action='store_true',help='All six models, tiny budgets; not reportable results')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--data-dir',type=Path,default=CRM_DIR,help='Folder with sales_pipeline/accounts/products/sales_teams CSVs')
    parser.add_argument('--feature-sets',default=None,help='Comma-separated subset of raw,history (default: both on data/crm, raw elsewhere)')
    args=parser.parse_args()
    sets=tuple(args.feature_sets.split(',')) if args.feature_sets else None
    run(args.output or default_output(args.data_dir,args.quick),quick=args.quick,data_dir=args.data_dir,feature_sets=sets)


if __name__=='__main__': main()
