import json
import numpy as np
import pandas as pd
import pytest


SIX={'dummy_prior','logistic_regression','random_forest','mlp','tabnet','catboost'}


@pytest.fixture(scope='module')
def completed_run(tmp_path_factory):
    from src.run_project import run
    output=tmp_path_factory.mktemp('complete_crm')
    manifest=run(output,quick=True,agent=False)
    return output,manifest


def test_runner_really_evaluates_all_models_and_scores_open_deals(completed_run):
    out,manifest=completed_run
    assert manifest['status']=='complete'
    assert {'train','evaluate','predict'}<=set(manifest['stages'])
    table=pd.read_csv(out/'metrics/test_metrics.csv')
    assert set(table.model)==SIX
    assert np.isfinite(table[['accuracy','precision','recall','f1','roc_auc','brier']]).all().all()
    scores=pd.read_csv(out/'predictions/open_deal_predictions.csv')
    assert len(scores)==1589
    assert scores.account_missing.sum()==1088
    assert np.allclose(scores.win_probability+scores.loss_probability,1)
    assert scores.explanation_method.eq('single_feature_reference_sensitivity').all()


def test_runner_audits_are_separate_and_reproducible(completed_run):
    out,manifest=completed_run
    from src.data.history import HISTORY_COLUMNS
    assert len(manifest['raw_sha256'])==5
    assert len(manifest['features'])==18+(len(HISTORY_COLUMNS) if manifest['feature_set']=='history' else 0)
    assert not set(['close_date','close_value','deal_stage','opportunity_id']) & set(manifest['features'])
    assert (pd.read_csv(out/'data/split_manifest.csv').role=='test').sum()==manifest['split']['test_rows']
    leak=pd.read_csv(out/'checks/leakage_audit.csv')
    assert leak.loc[leak.model.eq('LEAKY_close_value_stump'),'roc_auc'].iloc[0]>.99
    assert manifest['selected_model'] in set(pd.read_csv(out/'metrics/validation_metrics.csv').model)
    assert len(pd.read_csv(out/'metrics/calibration_test.csv'))==2
    assert json.loads((out/'explain/shap_audit.json').read_text())['max_additivity_error']<1e-5
    assert (out/'figures'/'roc_curves.png').stat().st_size>1000
    assert (out/'figures'/'reliability.png').stat().st_size>1000
    assert (out/'figures'/'precision_recall_curves.png').stat().st_size>1000
    for name in ['mlp','tabnet']:
        assert (out/'figures'/f'learning_curve_{name}.png').stat().st_size>1000
    assert (out/'models/model_bundle.joblib').stat().st_size>1000
    assert (out/'models/trained_models.joblib').stat().st_size>1000


def test_serialized_frozen_model_reproduces_export(completed_run):
    from src.data.crm import build
    from src.models.bundle import load_scoring
    out,_=completed_run
    from src.predict import add_history
    bundle=load_scoring(out/'models/model_bundle.joblib')
    rows=add_history(build().extra['scorable_open_deals'],bundle)
    p=bundle['model'].predict_proba(rows[bundle['features']])[:,1]
    exported=pd.read_csv(out/'predictions/open_deal_predictions.csv').set_index('opportunity_id')
    assert np.allclose(p,exported.loc[rows.opportunity_id,'win_probability'])


def test_stages_rerun_from_saved_models(completed_run,tmp_path):
    import shutil
    from src import evaluate,predict
    original,_=completed_run
    out=tmp_path/'rerun'; shutil.copytree(original,out)
    before=pd.read_csv(out/'metrics/test_metrics.csv')
    (out/'metrics/test_metrics.csv').unlink(); (out/'predictions/open_deal_predictions.csv').unlink()
    evaluate.run(out); predict.run(out)
    # Reloaded models must give exactly the metrics computed in memory during the chained run.
    pd.testing.assert_frame_equal(pd.read_csv(out/'metrics/test_metrics.csv'),before)
    assert len(pd.read_csv(out/'predictions/open_deal_predictions.csv'))==1589


def test_predict_scores_new_deals_in_pipeline_format(completed_run,tmp_path):
    from src.data.crm import CRM_DIR
    from src.predict import score_file
    out,_=completed_run
    raw=pd.read_csv(CRM_DIR/'sales_pipeline.csv')
    new=raw.loc[raw.deal_stage.eq('Engaging')&raw.account.notna(),['opportunity_id','sales_agent','product','account','engage_date']].head(5)
    new.to_csv(tmp_path/'new_deals.csv',index=False)
    scores=score_file(tmp_path/'new_deals.csv',out,tmp_path/'scores.csv')
    assert len(scores)==5 and set(scores.opportunity_id)==set(new.opportunity_id)
    assert scores.scoring_context.eq('user_supplied_rows').all() and not scores.account_missing.any()
    exported=pd.read_csv(out/'predictions/open_deal_predictions.csv').set_index('opportunity_id')
    assert np.allclose(scores.set_index('opportunity_id').win_probability,exported.loc[scores.opportunity_id,'win_probability'])
    bad=new.drop(columns='engage_date'); bad.to_csv(tmp_path/'bad.csv',index=False)
    with pytest.raises(ValueError,match='engage_date'): score_file(tmp_path/'bad.csv',out,tmp_path/'bad_scores.csv')


def test_evaluate_refuses_a_folder_without_trained_models(tmp_path):
    from src import evaluate
    with pytest.raises(RuntimeError,match='src.train'): evaluate.run(tmp_path)


def test_runner_measures_split_protocols_and_explanation_faithfulness(completed_run):
    out,manifest=completed_run
    summary=pd.read_csv(out/'checks/split_protocol_summary.csv')
    assert set(summary.protocol)=={'asof_purged','chronological_no_purge','random_stratified'}
    assert set(summary.model)==SIX-{'dummy_prior'}
    assert summary.loc[summary.protocol.eq('asof_purged'),'delta_vs_asof'].eq(0).all()
    detail=pd.read_csv(out/'checks/split_protocol_comparison.csv')
    asof=detail.loc[detail.protocol.eq('asof_purged')]
    assert asof.test_rows.eq(manifest['split']['test_rows']).all() and asof.train_rows.eq(manifest['split']['train_rows']).all()
    # Dropping the purge changes only training rows, never the test rows.
    assert detail.loc[detail.protocol.eq('chronological_no_purge'),'test_rows'].eq(manifest['split']['test_rows']).all()
    assert (detail.loc[detail.protocol.eq('chronological_no_purge'),'train_rows']>manifest['split']['train_rows']).all()
    assert (detail.auc_ci_low<=detail.roc_auc).all() and (detail.roc_auc<=detail.auc_ci_high).all()
    checks=json.loads((out/'checks/explanation_checks.json').read_text())
    agreement=checks['agreement']; deletion=checks['deletion']
    n=len(manifest['features'])
    assert agreement['features']==n and -1<=agreement['spearman_median']<=1
    assert 0<=agreement['top_overlap_mean']<=1 and agreement['top_overlap_random_expectation']==3/n
    assert deletion['rows']==manifest['split']['test_rows']
    assert deletion['mean_abs_change_top']>deletion['mean_abs_change_random']
    assert set(manifest['novelty_checks'])=={'split_protocol_max_auc_gain','rf_explanation_spearman_median','deletion_ratio_top_to_random'}


def test_feature_groups_cover_every_transformed_column():
    from src.data.crm import build
    from src.data.split import asof_split
    from src.evaluation.novelty_checks import feature_groups
    from src.models.zoo import fit_model
    d=build(); s=asof_split(d); rf=fit_model('random_forest',d,s.train,s.validation,quick=True)
    groups=feature_groups(rf,d)
    assert list(groups)==d.feature_columns
    positions=sorted(i for idx in groups.values() for i in idx)
    assert positions==list(range(len(rf.prep.get_feature_names_out())))
    names=rf.prep.get_feature_names_out()
    assert all(names[i].startswith('product_') for i in groups['product'])
    assert [names[i] for i in groups['sales_price']]==['sales_price']


def test_feature_sets_share_one_split_and_selection_is_locked_before_test(completed_run):
    out,manifest=completed_run
    table=pd.read_csv(out/'metrics/feature_set_comparison.csv')
    assert len(table)==12 and table.groupby('feature_set').size().to_dict()=={'history':6,'raw':6}
    assert np.isfinite(table[['validation_roc_auc','test_roc_auc','test_f1','test_brier']]).all().all()
    lock=json.loads((out/'models/selection_lock.json').read_text())
    assert lock['test_evaluated_at_selection'] is False and len(lock['candidates'])==12
    assert not any(k.startswith('test') for row in lock['candidates'] for k in row)
    best=table.sort_values(['validation_roc_auc','validation_brier','feature_set','model'],ascending=[False,True,True,True]).iloc[0]
    assert (lock['selected_feature_set'],lock['selected_model'])==(best.feature_set,best.model)
    assert (manifest['feature_set'],manifest['selected_model'])==(best.feature_set,best.model)
    assert table.selected.sum()==1 and table.loc[table.selected].iloc[0].model==best.model
    assert len(pd.read_csv(out/'data/history_coverage.csv'))==9
    # Each feature set's selected-set metrics are the same numbers the comparison reports.
    chosen=table.loc[table.feature_set.eq(manifest['feature_set'])].set_index('model')
    test=pd.read_csv(out/'metrics/test_metrics.csv').set_index('model')
    assert np.allclose(chosen.loc[test.index,'test_roc_auc'],test.roc_auc)


def test_model_card_uses_validation_only(completed_run):
    out,manifest=completed_run
    card=json.loads((out/'models/model_card.json').read_text())
    assert card['feature_set']==manifest['feature_set'] and card['validation_rows']==manifest['split']['validation_rows']
    low,high=card['validation_calibrated_roc_auc_ci95']; assert 0<=low<=high<=1
    assert set(card['committee_validation_quantiles'])==SIX-{'dummy_prior'}
    assert all(len(q)==101 and q==sorted(q) for q in card['committee_validation_quantiles'].values())
    assert 'Missing' not in card['training_categories']['sales_agent']
