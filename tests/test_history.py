import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from src.history import build_history, augment_dataset, HISTORY_COLUMNS
from src.datasets.crm import build
from src.temporal import asof_split


def archive():
    return pd.DataFrame([
        ['past','A','C','P','2017-01-01','2017-01-02',1,100.],
        ['same','A','C','P','2017-01-02','2017-01-04',0,0.],
        ['future','A','C','P','2017-01-02','2017-01-05',0,0.],
    ],columns=['opportunity_id','sales_agent','account','product','engage_date','close_date','is_won','close_value'])


def query(id='query', date='2017-01-04', account='C'):
    return pd.DataFrame([dict(opportunity_id=id,engage_date=date,sales_agent='A',account=account,product='P')],index=[17])


def test_strict_before_query_and_future_invariance():
    a=archive(); f,audit=build_history(query(),a)
    assert f.loc[17,'agent_closed_count']==1
    assert f.loc[17,'account_mean_close_value']==100
    assert f.loc[17,'product_mean_cycle_days']==1
    assert f.loc[17,'agent_win_rate']==1
    assert audit.loc[17,'latest_close_date']==pd.Timestamp('2017-01-02')
    a.loc[a.opportunity_id.isin(['same','future']),['is_won','close_value']]=[1,999999]
    g,_=build_history(query(),a)
    assert_frame_equal(f,g)


def test_self_exclusion_missing_keys_and_no_history():
    f,a=build_history(query('past'),archive())
    assert f.loc[17,'global_closed_count']==0
    assert f.loc[17,'account_win_rate']==.5
    f,_=build_history(query(account=np.nan),archive())
    assert f.loc[17,'account_closed_count']==0
    assert f.loc[17,'account_cold_start']==1
    assert f.loc[17,'account_win_rate']==1
    f,_=build_history(query(date='2016-01-01'),archive())
    assert f.loc[17,'global_closed_count']==0
    assert f.loc[17,'agent_mean_close_value']==0
    assert f.loc[17,'agent_mean_cycle_days']==0


def test_order_independent_and_inputs_unchanged():
    a=archive(); original=a.copy(deep=True)
    q=pd.concat([query(),query('second','2017-02-01').rename(index={17:25})])
    f,_=build_history(q,a)
    g,_=build_history(q.iloc[::-1],a.sample(frac=1,random_state=7))
    assert_frame_equal(f,g.loc[f.index]); assert_frame_equal(a,original)
    assert list(f)==HISTORY_COLUMNS


def test_frozen_train_archive_ignores_heldout_labels_and_audits():
    ds=build(); split=asof_split(ds); augmented=augment_dataset(ds,split)
    ds.frame.loc[np.r_[split.validation,split.test],'is_won'] ^= 1
    ds.frame.loc[np.r_[split.validation,split.test],'close_value']=999999
    changed=augment_dataset(ds,split)
    assert_frame_equal(augmented.features(),changed.features())
    assert set(augmented.extra['history_archive'].opportunity_id)==set(ds.frame.loc[split.train,'opportunity_id'])
    audit=augmented.extra['history_audit']
    assert (pd.to_datetime(audit.latest_close_date.dropna()) < pd.to_datetime(audit.loc[audit.latest_close_date.notna(),'query_date'])).all()
    assert len(augmented.extra['history_coverage'])==9
