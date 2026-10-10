import importlib
import numpy as np
import pandas as pd
import pytest

from src.data.crm import build


def splitter():
    assert importlib.util.find_spec('src.data.split') is not None, 'as-of split is not implemented'
    return importlib.import_module('src.data.split').asof_split


def test_asof_labels_are_known_before_next_period_and_dates_do_not_overlap():
    d = build()
    s = splitter()(d)
    f = d.frame
    assert len(s.train) > 100 and len(s.validation) > 100 and len(s.test) > 100
    assert f.loc[s.train, 'engage_date'].max() < s.validation_start
    assert f.loc[s.train, 'close_date'].max() < s.validation_start
    assert f.loc[s.validation, 'engage_date'].min() >= s.validation_start
    assert f.loc[s.validation, 'close_date'].max() < s.test_start
    assert f.loc[s.test, 'engage_date'].min() >= s.test_start
    assert not set(s.train) & set(s.validation) and not set(s.validation) & set(s.test)
    assert len(s.manifest) == 6711
    assert s.manifest['opportunity_id'].is_unique
    assert s.manifest['role'].isin(['train','validation','test','purged_train','purged_validation']).all()


def test_asof_partition_uses_dates_not_outcomes():
    d = build()
    s1 = splitter()(d)
    d.frame[d.target] = 1 - d.frame[d.target]
    s2 = splitter()(d)
    np.testing.assert_array_equal(s1.test, s2.test)


def test_asof_rejects_missing_dates():
    d = build()
    d.frame.loc[0,'close_date'] = pd.NaT
    with pytest.raises(ValueError, match='date'):
        splitter()(d)
