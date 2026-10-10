import pandas as pd
import pytest
from tests.test_project_run import completed_run
from src.assistant_demo import demo


def test_rule_assistant_demo_uses_real_bundle_and_matches_exports(completed_run):
    out,manifest=completed_run
    result=demo(out/'models/model_bundle.joblib')
    scores=pd.read_csv(out/'predictions/open_deal_predictions.csv').set_index('opportunity_id')
    identifier=result['opportunity']['opportunity_id']
    assert result['model_prediction']['win_probability']==pytest.approx(scores.loc[identifier,'win_probability'])
    assert result['timing']['scoring_context']=='post_model_engagement_snapshot'
    assert result['model']['feature_set']==manifest['feature_set']
    assert len(result['agent_recommendation']['actions'])>=2
    assert result['agent_recommendation']['autonomous_execution'] is False


def test_rule_assistant_demo_rejects_unknown_and_closed_ids(completed_run):
    from src.data.crm import build
    out,_=completed_run
    for identifier in ['DOES_NOT_EXIST',build().frame.opportunity_id.iloc[0]]:
        with pytest.raises(ValueError,match='Engaging'): demo(out/'models/model_bundle.joblib',identifier)
