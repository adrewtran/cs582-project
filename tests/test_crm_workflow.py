from pathlib import Path

from src.baselines import make_models, split
from src.crm_workflow import priority_group, score_open_deals
from src.datasets import crm


CRM_DATA = Path(__file__).resolve().parents[1] / "data" / "crm"


def test_priority_group_thresholds_are_explicit():
    assert priority_group(0.70) == "High"
    assert priority_group(0.69) == "Medium"
    assert priority_group(0.40) == "Medium"
    assert priority_group(0.39) == "Low"


def test_priority_group_rejects_invalid_probability():
    for invalid in (-0.01, 1.01):
        try:
            priority_group(invalid)
        except ValueError as error:
            assert "between 0 and 1" in str(error)
        else:
            raise AssertionError("invalid probability should be rejected")


def test_score_open_deals_returns_engaging_opportunities_and_probabilities():
    dataset = crm.build(CRM_DATA)
    X_train, _, y_train, _ = split(dataset, "temporal")
    model = make_models(dataset)["dummy_majority"].fit(X_train, y_train)

    scored = score_open_deals(model, dataset)

    assert len(scored) == 1589
    assert scored["opportunity_id"].notna().all()
    assert scored["win_probability"].between(0, 1).all()
    assert set(scored["priority"]) <= {"High", "Medium", "Low"}
    assert set(scored["deal_stage"]) == {"Engaging"}
