import json
import numpy as np
import pandas as pd
import pytest
from src.experiments import run_comparison


@pytest.fixture(scope='module')
def comparison(tmp_path_factory):
    out=tmp_path_factory.mktemp('comparison')
    result=run_comparison(out,quick=True,documents=False)
    return out,result


def test_comparison_six_models_same_split_and_locked_selection(comparison):
    out,manifest=comparison
    table=pd.read_csv(out/'ablation_comparison.csv')
    assert len(table)==12
    assert table.groupby('experiment').size().to_dict()=={'history':6,'raw':6}
    assert np.isfinite(table[['validation_roc_auc','test_roc_auc','test_f1','test_brier']]).all().all()
    a=pd.read_csv(out/'experiments/raw/split_manifest.csv')
    b=pd.read_csv(out/'experiments/history/split_manifest.csv')
    pd.testing.assert_frame_equal(a,b)
    lock=json.loads((out/'selection_lock.json').read_text())
    assert lock['test_evaluated_at_selection'] is False
    selected=table.sort_values(['validation_roc_auc','validation_brier','experiment','model'],ascending=[False,True,True,True]).iloc[0]
    assert lock['selected_experiment']==selected.experiment
    assert manifest['selected_model']==selected.model
    assert len(pd.read_csv(out/'history_coverage.csv'))==9
    assert manifest['status']=='complete'


def test_history_bundle_reproduces_export(comparison):
    import joblib
    from src.datasets.crm import build
    from src.history import build_history
    out,_=comparison
    bundle=joblib.load(out/'experiments/history/model_bundle.joblib')
    rows=build().extra['scorable_open_deals']
    history,_=build_history(rows,bundle['history_archive'])
    x=rows.join(history)[bundle['features']]
    p=bundle['model'].predict_proba(x)[:,1]
    scores=pd.read_csv(out/'experiments/history/open_deal_predictions.csv').set_index('opportunity_id')
    assert np.allclose(p,scores.loc[rows.opportunity_id,'win_probability'])
