"""Strictly prior closed-event aggregates, never whole-dataset target encoding.

The primary experiment freezes the archive to the training split. Date-only
events closing on the query date are excluded. Missing keys are not pooled.
"""
from collections import defaultdict
from dataclasses import replace
import numpy as np
import pandas as pd

ENTITIES={'agent':'sales_agent','account':'account','product':'product'}
STATS=['closed_count','win_rate','mean_close_value','mean_cycle_days','cold_start']
HISTORY_COLUMNS=[f'{entity}_{stat}' for entity in ENTITIES for stat in STATS]+[
    'global_closed_count','global_win_rate','global_mean_close_value','global_mean_cycle_days']
SMOOTHING=5.0
AUDIT_COLUMNS=['opportunity_id','query_date','latest_close_date','eligible_archive_count','archive_source','strict_before']


def _key(value):
    return None if pd.isna(value) or str(value).strip() in ('','Missing') else str(value)


def build_history(queries,archive):
    """Return features and row-level audit aligned exactly to queries.index."""
    if not queries.index.is_unique:
        raise ValueError('query index must be unique')
    required=['opportunity_id','engage_date',*ENTITIES.values()]
    if set(required)-set(queries) or set(required+['close_date','is_won','close_value'])-set(archive):
        raise ValueError('history columns missing')
    events=archive.copy(); q=queries[required].copy()
    if events.opportunity_id.isna().any() or events.opportunity_id.duplicated().any():
        raise ValueError('archive opportunity IDs must be unique and nonmissing')
    for frame,columns in [(events,['engage_date','close_date']),(q,['engage_date'])]:
        for col in columns:
            frame[col]=pd.to_datetime(frame[col],errors='raise')
            if frame[col].isna().any(): raise ValueError('history requires complete dates')
    if not events.is_won.isin([0,1]).all() or (events.close_date<events.engage_date).any():
        raise ValueError('invalid archive labels or cycle dates')
    events=events.sort_values(['close_date','opportunity_id'],kind='stable')
    # count, wins, value sum/count, cycle sum/count; missing values are not zeros.
    records=[]
    for row in events.to_dict('records'):
        value=float(row['close_value']) if pd.notna(row['close_value']) else np.nan
        cycle=(row['close_date']-row['engage_date']).days
        row['_vector']=np.array([1,row['is_won'],value if np.isfinite(value) else 0,
                                 int(np.isfinite(value)),cycle,1],dtype=float)
        records.append(row)
    global_state=np.zeros(6); states={e:defaultdict(lambda:np.zeros(6)) for e in ENTITIES}
    consumed={}; cursor=0; output={}; audits={}
    for idx,row in q.sort_values('engage_date',kind='stable').iterrows():
        while cursor<len(records) and records[cursor]['close_date']<row.engage_date:
            event=records[cursor]; v=event['_vector']; global_state+=v
            for entity,col in ENTITIES.items():
                key=_key(event[col])
                if key is not None: states[entity][key]+=v
            consumed[event['opportunity_id']]=event; cursor+=1
        own=consumed.get(row.opportunity_id)
        g=global_state.copy()
        if own is not None: g-=own['_vector']
        prior=g[1]/g[0] if g[0] else .5
        mean_value=g[2]/g[3] if g[3] else 0.
        mean_cycle=g[4]/g[5] if g[5] else 0.
        result={}
        for entity,col in ENTITIES.items():
            key=_key(row[col]); state=states[entity].get(key,np.zeros(6)).copy()
            if own is not None and key is not None and key==_key(own[col]): state-=own['_vector']
            result.update({f'{entity}_closed_count':state[0],
                f'{entity}_win_rate':(state[1]+SMOOTHING*prior)/(state[0]+SMOOTHING),
                f'{entity}_mean_close_value':state[2]/state[3] if state[3] else mean_value,
                f'{entity}_mean_cycle_days':state[4]/state[5] if state[5] else mean_cycle,
                f'{entity}_cold_start':int(state[0]==0)})
        result.update(global_closed_count=g[0],global_win_rate=prior,
                      global_mean_close_value=mean_value,global_mean_cycle_days=mean_cycle)
        latest=next((records[i]['close_date'] for i in range(cursor-1,-1,-1)
                     if records[i]['opportunity_id']!=row.opportunity_id),pd.NaT)
        output[idx]=result
        audits[idx]={'opportunity_id':row.opportunity_id,'query_date':row.engage_date,
            'latest_close_date':latest,'eligible_archive_count':int(g[0]),
            'archive_source':'frozen_train_only','strict_before':True}
    features=pd.DataFrame.from_dict(output,orient='index',columns=HISTORY_COLUMNS).reindex(q.index)
    audit=pd.DataFrame.from_dict(audits,orient='index',columns=AUDIT_COLUMNS).reindex(q.index)
    return features.astype(float),audit


def augment_dataset(dataset,split):
    archive=dataset.frame.loc[split.train].copy()
    features,audit=build_history(dataset.frame,archive)
    frame=dataset.frame.join(features)
    extra=dict(dataset.extra); extra['history_archive']=archive
    extra['history_audit']=audit.join(split.manifest[['role']])
    coverage=[]
    for role,idx in [('train',split.train),('validation',split.validation),('test',split.test)]:
        for entity in ENTITIES:
            counts=features.loc[idx,f'{entity}_closed_count']
            coverage.append({'split':role,'entity':entity,'rows':len(idx),
                'with_history':int((counts>0).sum()),'coverage':float((counts>0).mean()),
                'median_prior_count':float(counts.median())})
    extra['history_coverage']=pd.DataFrame(coverage)
    open_rows=extra['scorable_open_deals']
    open_features,open_audit=build_history(open_rows,archive)
    extra['scorable_open_deals']=open_rows.join(open_features)
    extra['open_history_audit']=open_audit
    return replace(dataset,frame=frame,numeric=dataset.numeric+HISTORY_COLUMNS,extra=extra)
