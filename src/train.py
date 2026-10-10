"""Train stage: fit all five models, select on validation, calibrate, save the models.

    python -m src.train [--quick] [--output PATH]

No test row is scored here: selection, threshold and calibration are locked before evaluation.
"""
import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from importlib.metadata import version
import platform
import subprocess
import pandas as pd
from src.data.crm import build,CRM_DIR
from src.data.split import asof_split
from src.evaluation.metrics import metrics,choose_threshold,select_model
from src.explain.reference import make_reference
from src.models import bundle
from src.models.calibration import CalibratedModel
from src.models.zoo import MODEL_NAMES,fit_model
from src.outputs import ROOT,DEFAULT_OUTPUT,SMOKE_OUTPUT,Outputs,write_json

DEPENDENCIES=['numpy','pandas','scipy','scikit-learn','torch','pytorch-tabnet','shap','matplotlib','joblib']


def describe_run(manifest,dataset,split):
    y=dataset.labels()
    manifest['features']=dataset.feature_columns
    manifest['raw_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(CRM_DIR.glob('*.csv'))}
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


def run(output_dir=DEFAULT_OUTPUT,quick=False,dataset=None,split=None):
    """Returns (manifest, trained, dataset, split) so run_project can chain stages in memory."""
    out=Outputs(output_dir).make()
    manifest={'status':'running','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full',
              'started_utc':datetime.now(timezone.utc).isoformat(),'seed':42,'python':platform.python_version(),
              'platform':platform.platform(),'machine':platform.machine(),
              'run_location':'local CPU execution; see platform and machine'}
    out.write_manifest(manifest)
    dataset=dataset or build(); split=split or asof_split(dataset)
    split.manifest.to_csv(out.data/'split_manifest.csv',index=False)
    write_json(out.data/'data_quality.json',dataset.report)
    describe_run(manifest,dataset,split)
    X=dataset.features(); xv,yv=X.loc[split.validation],dataset.labels().loc[split.validation]
    models={}; thresholds={}; rows=[]; histories={}
    for name in MODEL_NAMES:
        print(f'Training {name} on CPU...',flush=True)
        model=fit_model(name,dataset,split.train,split.validation,quick=quick); models[name]=model
        pv=model.predict_proba(xv)[:,1]; thresholds[name]=choose_threshold(yv,pv)
        rows.append({'model':name,**metrics(yv,pv,thresholds[name])})
        histories[name]={'settings':model.settings,'history':model.history,'fit_seconds':model.fit_seconds}
    selected=select_model(rows)
    calibrated=CalibratedModel.fit(models[selected],xv,yv)
    reference=make_reference(X.loc[split.train],dataset)
    trained=bundle.Trained(models=models,thresholds=thresholds,selected=selected,calibrated=calibrated,
        decision_threshold=float(calibrated.transform(thresholds[selected])),reference=reference,
        features=dataset.feature_columns,validation_rows=rows)
    pd.DataFrame(rows).to_csv(out.metrics/'validation_metrics.csv',index=False)
    write_json(out.models/'model_training.json',histories)
    write_json(out.models/'explanation_reference.json',reference)
    bundle.save(out,trained)
    manifest.update(selected_model=selected,selection_rule='validation ROC-AUC, then lower Brier, then model name',
        calibration={'method':'positive-slope sigmoid on validation logit scores; base remains frozen',
                     'slope':calibrated.slope,'intercept':calibrated.intercept},
        raw_decision_threshold=thresholds[selected],calibrated_decision_threshold=trained.decision_threshold,
        threshold_rule='validation macro-F1; 0.10..0.90 by 0.01; ties closest to 0.50',
        priority={'medium_min':.4,'high_min':.7,'status':'unvalidated heuristic, not a business policy'},
        status='trained',stages={'train':datetime.now(timezone.utc).isoformat()})
    out.write_manifest(manifest)
    print(f'TRAINED: selected={selected}; models saved in {out.models}',flush=True)
    return manifest,trained,dataset,split


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick',action='store_true',help='All five models, tiny budgets; not reportable results')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    run(args.output or (SMOKE_OUTPUT if args.quick else DEFAULT_OUTPUT),quick=args.quick)


if __name__=='__main__': main()
