from pathlib import Path

from src import baselines
from src.datasets import crm


CRM_DATA = Path(__file__).resolve().parents[1] / "data" / "crm"


def test_crm_temporal_split_uses_later_rows_for_test():
    dataset = crm.build(CRM_DATA)

    X_train, X_test, y_train, y_test = baselines.split(dataset, "temporal")

    assert len(X_train) + len(X_test) == len(dataset.frame)
    assert X_train.index.max() < X_test.index.min()
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_baseline_models_include_majority_reference():
    dataset = crm.build(CRM_DATA)

    assert set(baselines.make_models(dataset)) == {
        "dummy_majority",
        "logistic_regression",
        "random_forest",
    }


def test_crm_is_the_default_dataset():
    assert baselines.DEFAULT_DATASET == "crm"


def test_dummy_model_has_no_feature_importance():
    dataset = crm.build(CRM_DATA)
    model = baselines.make_models(dataset)["dummy_majority"]
    X_train, _, y_train, _ = baselines.split(dataset, "temporal")
    model.fit(X_train, y_train)

    assert baselines.feature_importance("dummy_majority", model) is None
