"""A/B/C experiment: freeze all validation choices before consulting test metrics."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import time
import pandas as pd
from src.datasets.crm import build
from src.history import augment_dataset
from src.temporal import asof_split
from src.run_project import prepare,run,write_json,DEFAULT_OUTPUT


def run_comparison(output_dir=DEFAULT_OUTPUT,quick=False,documents=True):
    start=time.perf_counter()
    started_utc=datetime.now(timezone.utc).isoformat()
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    manifest={'status':'running','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full',
              'started_utc':started_utc}
    write_json(out/'run_manifest.json',manifest)
    try:
        raw=build(); split=asof_split(raw)
        datasets={'raw':raw,'history':augment_dataset(raw,split)}
        prepared={name:prepare(ds,split,quick) for name,ds in datasets.items()}
        candidates=[dict(experiment=name,**row) for name,p in prepared.items() for row in p['val_rows']]
        selected=min(candidates,key=lambda r:(-r['roc_auc'],r['brier'],r['experiment'],r['model']))
        lock={'selected_experiment':selected['experiment'],'selected_model':selected['model'],
              'locked_utc':datetime.now(timezone.utc).isoformat(),'test_evaluated_at_selection':False,
              'selection_rule':'validation ROC-AUC, lower validation Brier, experiment, model name',
              'candidates':candidates,'configurations':{k:v['histories'] for k,v in prepared.items()},
              'history_protocol':'frozen training archive; close_date strictly before engagement; exclude query ID',
              'warning':'Previously inspected exploratory test, not pristine; no post-test tuning.'}
        write_json(out/'selection_lock.json',lock)
        results={}; combined=[]
        for name,ds in datasets.items():
            target=out/'experiments'/name
            results[name]=run(target,quick,False,dataset=ds,split=split,prepared=prepared[name])
            val=pd.read_csv(target/'validation_metrics.csv').set_index('model').add_prefix('validation_')
            test=pd.read_csv(target/'test_metrics.csv').set_index('model').add_prefix('test_')
            table=val.join(test).reset_index(); table.insert(0,'experiment',name); combined.append(table)
        comparison=pd.concat(combined,ignore_index=True)
        comparison.to_csv(out/'ablation_comparison.csv',index=False)
        delta=comparison.pivot(index='model',columns='experiment',values='test_roc_auc')
        delta['history_minus_raw']=delta['history']-delta['raw']
        delta.to_csv(out/'history_auc_deltas.csv')
        chosen=selected['experiment']; source=out/'experiments'/chosen
        for path in source.iterdir():
            if path.is_dir(): shutil.copytree(path,out/path.name,dirs_exist_ok=True)
            else: shutil.copy2(path,out/path.name)
        for name in ['history_audit','history_coverage','open_history_audit']:
            shutil.copy2(out/'experiments/history'/f'{name}.csv',out/f'{name}.csv')
        manifest=results[chosen].copy()
        manifest.update(status='running',selected_experiment=chosen,comparison_models=12,
                        selection_lock='selection_lock.json',history_protocol=lock['history_protocol'],
                        selected_variant_export_seconds=manifest['elapsed_seconds'],
                        started_utc=started_utc,elapsed_seconds=time.perf_counter()-start,
                        timing_scope='entire raw/history comparison, training and deliverables')
        write_json(out/'run_manifest.json',manifest)
        # Enrichment is independent of whether the validation winner used histories.
        from src.sales_agent import export_agent_outputs
        export_agent_outputs(out,datasets['history'])
        if documents:
            from src.upgrade_deliverables import build_deliverables
            build_deliverables(out)
        manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),
                        elapsed_seconds=time.perf_counter()-start)
        write_json(out/'run_manifest.json',manifest)
        print(f'COMPARISON COMPLETE: {chosen}/{selected["model"]}; {out}',flush=True)
        return manifest
    except Exception as exc:
        manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}',
                        elapsed_seconds=time.perf_counter()-start)
        write_json(out/'run_manifest.json',manifest)
        raise
