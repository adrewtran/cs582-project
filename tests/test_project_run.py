import json
import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def completed_run(tmp_path_factory):
    from src.run_project import run
    output=tmp_path_factory.mktemp('complete_crm')
    manifest=run(output,quick=True,documents=False)
    return output,manifest


def test_runner_really_evaluates_all_models_and_scores_open_deals(completed_run):
    out,manifest=completed_run
    assert manifest['status']=='complete'
    table=pd.read_csv(out/'test_metrics.csv')
    assert set(table.model)=={'dummy_prior','logistic_regression','random_forest','mlp','tabnet'}
    assert np.isfinite(table[['accuracy','precision','recall','f1','roc_auc','brier']]).all().all()
    scores=pd.read_csv(out/'open_deal_predictions.csv')
    assert len(scores)==1589
    assert scores.account_missing.sum()==1088
    assert np.allclose(scores.win_probability+scores.loss_probability,1)
    assert scores.explanation_method.eq('single_feature_reference_sensitivity').all()


def test_runner_audits_are_separate_and_reproducible(completed_run):
    out,manifest=completed_run
    assert len(manifest['raw_sha256'])==5
    assert len(manifest['features'])==18
    assert not set(['close_date','close_value','deal_stage','opportunity_id']) & set(manifest['features'])
    assert (pd.read_csv(out/'split_manifest.csv').role=='test').sum()==manifest['split']['test_rows']
    leak=pd.read_csv(out/'leakage_audit.csv')
    assert leak.loc[leak.model.eq('LEAKY_close_value_stump'),'roc_auc'].iloc[0]>.99
    assert manifest['selected_model'] in set(pd.read_csv(out/'validation_metrics.csv').model)
    assert len(pd.read_csv(out/'calibration_test.csv'))==2
    assert json.loads((out/'shap_audit.json').read_text())['max_additivity_error']<1e-5
    assert (out/'figures'/'roc_curves.png').stat().st_size>1000
    assert (out/'figures'/'reliability.png').stat().st_size>1000
    assert (out/'figures'/'precision_recall_curves.png').stat().st_size>1000
    for name in ['mlp','tabnet']:
        assert (out/'figures'/f'learning_curve_{name}.png').stat().st_size>1000
    assert (out/'model_bundle.joblib').stat().st_size>1000


def test_serialized_frozen_model_reproduces_export(completed_run):
    import joblib
    from src.datasets.crm import build
    out,_=completed_run
    bundle=joblib.load(out/'model_bundle.joblib')
    dataset=build(); rows=dataset.extra['scorable_open_deals']
    p=bundle['model'].predict_proba(rows[dataset.feature_columns])[:,1]
    exported=pd.read_csv(out/'open_deal_predictions.csv').set_index('opportunity_id')
    assert np.allclose(p,exported.loc[rows.opportunity_id,'win_probability'])
