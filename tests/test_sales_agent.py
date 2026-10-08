import math
import pytest
from src.sales_agent import recommend


def test_deterministic_risk_and_evidence_grounded_actions():
    history={'account_closed_count':0,'agent_closed_count':10,'agent_win_rate':.2,
             'product_closed_count':10,'product_win_rate':.3,'global_win_rate':.7}
    r=recommend({'account':None,'product':'P'},.2,history,{'sales_price':-.04},[])
    assert r==recommend({'account':None,'product':'P'},.2,history,{'sales_price':-.04},[])
    assert r['model_prediction']['loss_probability']==.8
    agent=r['agent_recommendation']; assert agent['loss_risk']=='HIGH'
    assert 2<=len(agent['actions'])<=4
    assert len({a['rule_id'] for a in agent['actions']})==len(agent['actions'])
    assert 'complete_account' in [a['rule_id'] for a in agent['actions']]
    assert all(a['evidence'] and a['rationale'] for a in agent['actions'])
    assert agent['causal_claim'] is False


def test_no_supported_history_no_underperformance_claim_and_boundary_risk():
    for p,risk in [(.399,'HIGH'),(.4,'MEDIUM'),(.699,'MEDIUM'),(.7,'LOW')]:
        result=recommend({'account':'A'},p,{}, {}, [])['agent_recommendation']
        assert result['loss_risk']==risk
        assert not {'product_review','senior_review'} & {a['rule_id'] for a in result['actions']}
    for invalid in [-.1,1.1,math.nan]:
        with pytest.raises(ValueError): recommend({},invalid,{}, {}, [])


def test_supported_product_review_includes_observed_counts():
    result=recommend({'account':'A'},.6,{'account_closed_count':8,'product_closed_count':12,
        'product_win_rate':.2,'global_win_rate':.6}, {}, [])
    action=next(a for a in result['agent_recommendation']['actions'] if a['rule_id']=='product_review')
    assert action['evidence']['prior_closed_count']==12
    assert action['evidence']['smoothed_win_rate']==.2
