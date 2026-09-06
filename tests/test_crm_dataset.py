from pathlib import Path

import pandas as pd

from src.datasets import crm


CRM_DATA = Path(__file__).resolve().parents[1] / "data" / "crm"


def test_product_name_is_normalized_before_join():
    pipeline = pd.DataFrame(
        {
            "product": ["GTXPro"],
            "engage_date": ["1/2/17"],
            "close_date": ["1/3/17"],
        }
    )

    cleaned = crm.clean_pipeline(pipeline)

    assert cleaned.loc[0, "product"] == "GTX Pro"


def test_crm_build_preserves_expected_rows_and_targets():
    dataset = crm.build(CRM_DATA)

    assert len(dataset.frame) == 6711
    assert dataset.report["open_rows"] == 2089
    assert dataset.report["positives"] == 4238
    assert dataset.report["negatives"] == 2473


def test_crm_honest_features_exclude_all_leakage_columns():
    dataset = crm.build(CRM_DATA)

    forbidden = {"deal_stage", "close_date", "close_value", "opportunity_id"}
    assert forbidden.isdisjoint(dataset.feature_columns)
    assert forbidden.isdisjoint(dataset.features().columns)


def test_crm_labeled_rows_are_chronological_for_temporal_holdout():
    dataset = crm.build(CRM_DATA)

    assert dataset.report["row_order_is_temporal"] is True
    assert dataset.frame["engage_date"].is_monotonic_increasing


def test_only_engaging_rows_with_dates_are_scorable():
    dataset = crm.build(CRM_DATA)
    scorable = dataset.extra["scorable_open_deals"]

    assert not scorable.empty
    assert set(scorable["deal_stage"]) == {"Engaging"}
    assert scorable["engage_date"].notna().all()
    assert len(scorable) == 1589
