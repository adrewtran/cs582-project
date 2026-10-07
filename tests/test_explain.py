import importlib
from dataclasses import replace
import numpy as np
import pytest
from src.datasets.crm import build
from src.baselines import make_models,split
from src.crm_workflow import score_open_deals


def test_legacy_scoring_accepts_empty_open_set():
    d=build(); x,_,y,_=split(d,'temporal'); m=make_models(d)['logistic_regression'].fit(x,y)
    empty=replace(d,extra=d.extra|{'scorable_open_deals':d.extra['scorable_open_deals'].iloc[:0]})
    result=score_open_deals(m,empty,include_explanations=True)
    assert result.empty and 'negative_factors' in result


def test_legacy_lr_factors_can_reconstruct_log_odds_including_intercept():
    d=build(); x,_,y,_=split(d,'temporal'); m=make_models(d)['logistic_regression'].fit(x,y)
    scores=score_open_deals(m,d,include_explanations=True)
    assert 'model_log_odds' in scores and 'intercept_log_odds' in scores and 'sum_feature_log_odds' in scores
    np.testing.assert_allclose(scores.model_log_odds,scores.intercept_log_odds+scores.sum_feature_log_odds)
    np.testing.assert_allclose(1/(1+np.exp(-scores.model_log_odds)),scores.win_probability)


def test_reference_explanations_are_labeled_and_flag_missing_accounts():
    assert importlib.util.find_spec('src.explain') is not None,'explanations are not implemented'
    from src.explain import explain_open,make_reference
    from src.models import fit_model
    from src.temporal import asof_split
    d=build(); s=asof_split(d); m=fit_model('logistic_regression',d,s.train,s.validation,quick=True)
    r=make_reference(d.features().loc[s.train],d)
    out=explain_open(m,d,r,.5)
    assert len(out)==1589 and out.opportunity_id.is_unique
    assert out.account_missing.sum()==1088
    assert set(out.explanation_method)=={'single_feature_reference_sensitivity'}
    assert set(out.scoring_context)=={'frozen_model_snapshot_demo'}
    np.testing.assert_allclose(out.win_probability+out.loss_probability,1)
    assert out.positive_factors.notna().all() and out.negative_factors.notna().all()
