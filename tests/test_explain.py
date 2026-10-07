import importlib
import numpy as np
from src.datasets.crm import build


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
