import importlib
import numpy as np
import pytest
from src.data.crm import build
from src.data.split import asof_split


def module(name):
    assert importlib.util.find_spec(name) is not None, f'{name} is not implemented'
    return importlib.import_module(name)


@pytest.mark.parametrize('name',['dummy_prior','logistic_regression','random_forest','mlp','tabnet','catboost'])
def test_all_six_models_fit_real_data_and_return_complementary_probabilities(name):
    d=build(); s=asof_split(d)
    m=module('src.models.zoo').fit_model(name,d,s.train,s.validation,quick=True)
    p=m.predict_proba(d.features().loc[s.test[:12]])
    assert p.shape == (12,2)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    np.testing.assert_allclose(p.sum(axis=1),1)
    assert m.train_rows == len(s.train)
    assert m.fit_seconds >= 0
    assert m.predict_proba(d.features().iloc[:0]).shape == (0,2)


def test_metrics_match_hand_computed_example_and_reject_bad_probabilities():
    e=module('src.evaluation.metrics')
    row=e.metrics([0,0,1,1],[.1,.8,.7,.9],.5)
    assert row['accuracy'] == .75
    assert row['recall'] == 1
    assert row['precision'] == pytest.approx(2/3)
    assert row['f1'] == pytest.approx(.8)
    assert row['roc_auc'] == .75
    assert (row['tn'],row['fp'],row['fn'],row['tp']) == (1,1,0,2)
    with pytest.raises(ValueError): e.metrics([0,1],[.2,np.nan])
    with pytest.raises(ValueError): e.metrics([0,0],[.2,.3])


def test_threshold_tie_breaking_and_frozen_monotone_calibration():
    e=module('src.evaluation.metrics')
    assert e.choose_threshold([0,0,1,1],[.1,.2,.8,.9]) == pytest.approx(.5)
    d=build(); s=asof_split(d)
    m=module('src.models.zoo').fit_model('logistic_regression',d,s.train,s.validation,quick=True)
    x=d.features().loc[s.test[:10]]; before=m.predict_proba(x).copy()
    c=module('src.models.calibration').CalibratedModel.fit(m,d.features().loc[s.validation],d.labels().loc[s.validation])
    after=c.predict_proba(x)
    np.testing.assert_allclose(before,m.predict_proba(x))
    np.testing.assert_allclose(after[:,1],c.transform(before[:,1]))
    assert np.all(np.diff(c.transform(np.linspace(.01,.99,100)))>=0)


def test_model_selection_uses_validation_table_and_deterministic_ties():
    e=module('src.evaluation.metrics')
    rows=[{'model':'b','roc_auc':.7,'brier':.2},{'model':'a','roc_auc':.7,'brier':.1}]
    assert e.select_model(rows) == 'a'
