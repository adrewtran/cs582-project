import json
import pandas as pd
import pytest
from tests.test_experiments import comparison
from src.demo import demo


def test_demo_uses_real_bundle_and_matches_exports(comparison):
    out,manifest=comparison
    result=demo(out/'model_bundle.joblib')
    scores=pd.read_csv(out/'open_deal_predictions.csv').set_index('opportunity_id')
    identifier=result['opportunity']['opportunity_id']
    assert result['model_prediction']['win_probability']==pytest.approx(scores.loc[identifier,'win_probability'])
    saved=json.loads((out/'demo_example.json').read_text())
    assert saved['opportunity']['opportunity_id']==identifier
    assert saved['model_prediction']['win_probability']==pytest.approx(result['model_prediction']['win_probability'])
    assert result['timing']['scoring_context']=='post_model_engagement_snapshot'
    assert len(result['agent_recommendation']['actions'])>=2


def test_demo_rejects_unknown_and_closed_ids(comparison):
    from src.datasets.crm import build
    out,_=comparison
    for identifier in ['DOES_NOT_EXIST',build().frame.opportunity_id.iloc[0]]:
        with pytest.raises(ValueError,match='Engaging'): demo(out/'model_bundle.joblib',identifier)
