import pandas as pd
import pytest
from src.datasets import crm


@pytest.mark.parametrize('change,expected', [
    ('bad_date','date'), ('bad_stage','stage'), ('duplicate','opportunity_id'),
    ('missing_product','product'), ('close_before_engage','close'), ('bad_numeric','employees'),
])
def test_raw_validation_rejects_corrupt_records(tmp_path,change,expected):
    raw=crm.load_raw()
    p=raw['sales_pipeline']
    closed=p.index[p['deal_stage'].eq('Won')][0]
    if change=='bad_date': p.loc[closed,'engage_date']='not-a-date'
    elif change=='bad_stage': p.loc[closed,'deal_stage']='Winning'
    elif change=='duplicate': p.loc[1,'opportunity_id']=p.loc[0,'opportunity_id']
    elif change=='missing_product': p.loc[closed,'product']='nonexistent'
    elif change=='close_before_engage': p.loc[closed,'close_date']='1/1/00'
    elif change=='bad_numeric': raw['accounts']['employees']=raw['accounts']['employees'].astype(object); raw['accounts'].loc[0,'employees']='bad'
    for name,frame in raw.items(): frame.to_csv(tmp_path/f'{name}.csv',index=False)
    with pytest.raises(ValueError,match=expected): crm.build(tmp_path)


def test_report_exposes_missing_accounts_without_dropping_open_deals():
    d=crm.build()
    assert d.report.get('scorable_missing_account') == 1088
    assert d.report.get('product_name_fixes') == 1480
    assert len(d.extra['scorable_open_deals']) == 1589
