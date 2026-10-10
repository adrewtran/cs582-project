"""Local, inspectable tools the agent calls through its controller. No network, no external system.

Every tool returns a ToolResult with the data it observed and the exact sources (table, key, fields) so a
reviewer can re-verify each claim. Outcome fields (close_date, close_value, final deal_stage of a closed
deal) are never returned: the agent cannot see the label it is reasoning about.
"""
from dataclasses import dataclass,field
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from src.agent import policy
from src.data.crm import CATEGORICAL,SECTOR_FIXES,build,clean_accounts,load_raw
from src.data.history import HISTORY_COLUMNS,build_history
from src.models.bundle import SCORING_FILE,load_scoring,load_trained
from src.outputs import Outputs

OBSERVABLE=['opportunity_id','sales_agent','product','account','engage_date']
ACCOUNT_FIELDS=['sector','year_established','revenue','employees','office_location','subsidiary_of']
FORBIDDEN=('close_date','close_value','is_won')


@dataclass
class Tool:
    name: str
    permission: str           # read | compute | local_write | external_write | communication
    provides: str             # belief slot the result fills ('' for actions)
    description: str
    fn: object


@dataclass
class ToolResult:
    status: str               # ok | error
    data: dict=field(default_factory=dict)
    sources: list=field(default_factory=list)
    error: str=''


def jsonable(value):
    """Plain JSON types only, so traces hash and compare deterministically."""
    if isinstance(value,dict): return {str(k):jsonable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)): return [jsonable(v) for v in value]
    if isinstance(value,(pd.Timestamp,)): return None if pd.isna(value) else str(value.date())
    if isinstance(value,(np.integer,)): return int(value)
    if isinstance(value,(np.floating,float)):
        value=float(value); return None if math.isnan(value) else round(value,10)
    if isinstance(value,np.bool_): return bool(value)
    if value is None or isinstance(value,(str,int,bool)): return value
    if pd.isna(value): return None
    return str(value)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Resources:
    """Everything the tools may read: the frozen run artifacts and the raw CRM tables (read-only)."""

    def __init__(self,run_dir,faults=None):
        self.out=Outputs(run_dir)
        manifest=self.out.read_manifest()
        if 'selected_model' not in manifest: raise RuntimeError(f'{self.out.root} has no trained models: run python -m src.run_project first')
        self.scoring=load_scoring(self.out.models/SCORING_FILE)
        self.trained=load_trained(self.out)
        self.card=json.loads((self.out.models/'model_card.json').read_text())
        self.bundle_sha256=file_sha256(self.out.models/SCORING_FILE)
        data_dir=self.out.source_data()
        dataset=build(data_dir)
        self.closed=dataset.frame.set_index('opportunity_id',drop=False).rename_axis(None)
        self.open=dataset.extra['scorable_open_deals'].set_index('opportunity_id',drop=False).rename_axis(None)
        self.not_scorable=dataset.extra['open_deals'].loc[lambda f:~f.opportunity_id.isin(self.open.index)].set_index('opportunity_id',drop=False).rename_axis(None)
        raw_accounts=load_raw(data_dir)['accounts']
        self.raw_sectors=raw_accounts.set_index('account').sector.to_dict()
        self.accounts=clean_accounts(raw_accounts).set_index('account',drop=False).rename_axis(None)
        self.archive=self.scoring['history_archive']
        cycles=(self.archive.close_date-self.archive.engage_date).dt.days
        self.stale_days=float(cycles.quantile(policy.STALE_QUANTILE))
        self.snapshot_date=max(dataset.frame.close_date.max(),dataset.frame.engage_date.max(),self.open.engage_date.max())
        test=dataset.frame.loc[dataset.frame.engage_date>=pd.Timestamp(self.scoring['model_available_date'])]
        # Outcome-masked replay pool: labelled test-period deals with every outcome column removed.
        self.replay=test.drop(columns=list(FORBIDDEN)+['deal_stage']).assign(deal_stage='masked_for_replay').set_index('opportunity_id',drop=False).rename_axis(None)
        self.faults=dict(faults or {})
        self.manifest=manifest
        self._inputs={}; self._history={}

    def row(self,opportunity_id,pool):
        table=self.open if pool=='open' else self.replay
        return table.loc[opportunity_id]

    def model_inputs(self,opportunity_id,pool):
        """The exact feature row the frozen model scores: the training pipeline's join + strictly prior history."""
        key=(opportunity_id,pool)
        if key not in self._inputs:
            frame=(self.open if pool=='open' else self.replay).loc[[opportunity_id]].reset_index(drop=True)
            if self.scoring['feature_set']=='history':
                frame=frame.join(self.history(opportunity_id,pool)[0])
            self._inputs[key]=frame[self.scoring['features']]
        return self._inputs[key].copy()

    def history(self,opportunity_id,pool):
        """Strictly prior aggregates for one deal from the frozen training archive (cached per deal)."""
        key=(opportunity_id,pool)
        if key not in self._history:
            row=self.row(opportunity_id,pool)
            query=pd.DataFrame([row[['opportunity_id','engage_date','sales_agent','account','product']]]).reset_index(drop=True)
            self._history[key]=build_history(query,self.archive)
        return self._history[key]


def source(table,key,fields):
    return {'table':table,'key':key,'fields':list(fields)}


# ---------------------------------------------------------------------------------- read tools

def get_opportunity(res,opportunity_id,pool='open'):
    if not isinstance(opportunity_id,str) or not opportunity_id.strip():
        return ToolResult('error',error='invalid_input: opportunity_id must be a nonempty string')
    if pool not in ('open','test_replay'): return ToolResult('error',error=f'invalid_input: unknown pool {pool}')
    table=res.open if pool=='open' else res.replay
    if opportunity_id not in table.index:
        if pool=='open' and opportunity_id in res.closed.index:
            return ToolResult('error',error='not_eligible: closed deal; only dated Engaging opportunities are scored')
        if pool=='open' and opportunity_id in res.not_scorable.index:
            return ToolResult('error',error='not_eligible: Prospecting deal without an engage_date')
        return ToolResult('error',error=f'not_found: {opportunity_id} is not in the {pool} pool')
    row=table.loc[opportunity_id]
    data={k:jsonable(row[k]) for k in OBSERVABLE}
    data['deal_stage']=str(row['deal_stage'])
    data['days_open_at_snapshot']=int((res.snapshot_date-row.engage_date).days)
    data['snapshot_date']=str(res.snapshot_date.date())
    return ToolResult('ok',data,[source('data/crm/sales_pipeline.csv',opportunity_id,OBSERVABLE+['deal_stage'])])


def get_account_information(res,account):
    if account is None: return ToolResult('error',error='invalid_input: no account key on the opportunity')
    if account not in res.accounts.index: return ToolResult('error',error=f'not_found: account {account!r} is not in accounts.csv')
    row=res.accounts.loc[account]
    data={'account':account,**{k:jsonable(row[k]) for k in ACCOUNT_FIELDS}}
    raw_sector=next((typo for typo,fixed in SECTOR_FIXES.items() if fixed==row['sector'] and res.raw_sectors.get(account)==typo),None)
    if raw_sector: data['normalized']={'sector':f'{raw_sector} -> {row["sector"]} (documented spelling fix)'}
    return ToolResult('ok',data,[source('data/crm/accounts.csv',account,ACCOUNT_FIELDS)])


def get_prior_sales_history(res,opportunity_id,pool='open'):
    history,audit=res.history(opportunity_id,pool)
    h=history.iloc[0]; a=audit.iloc[0]
    data={k:jsonable(h[k]) for k in HISTORY_COLUMNS}
    data.update(eligible_archive_count=int(a.eligible_archive_count),strict_before=bool(a.strict_before),
                latest_prior_close_date=jsonable(a.latest_close_date),query_date=jsonable(a.query_date),
                archive='frozen training archive (closed strictly before engagement; own outcome excluded)')
    return ToolResult('ok',data,[source('models/model_bundle.joblib:history_archive',opportunity_id,
        ['sales_agent','account','product','engage_date','close_date','is_won','close_value'])])


def check_data_quality(res,opportunity_id,pool='open',account_found=None):
    """Issues are 'blocking' (the score is outside training support) or 'warning' (use with care)."""
    row=res.row(opportunity_id,pool); X=res.model_inputs(opportunity_id,pool).iloc[0]
    issues=[]
    def issue(code,severity,field,detail): issues.append({'code':code,'severity':severity,'field':field,'detail':detail})
    if pd.isna(row['account']):
        issue('account_missing','blocking','account','No account: account inputs are imputed, outside the complete-account training support.')
    elif account_found is False:
        issue('account_not_found','blocking','account','Account key does not match accounts.csv.')
    imputed=[c for c in res.scoring['features'] if pd.isna(X[c])]
    if imputed and not pd.isna(row['account']):
        issue('imputed_inputs','warning',','.join(imputed),'Model inputs will be imputed with training medians/modes.')
    for col in CATEGORICAL:
        value=X[col]
        if not pd.isna(value) and str(value) not in res.card['training_categories'][col]:
            issue('unseen_category','warning',col,f'{value!r} never appeared in training rows.')
    for col,(low,high) in res.card['training_numeric_range'].items():
        # History counts for deals engaged after training can exceed anything the model saw: flag the extrapolation.
        value=X.get(col)
        if value is not None and not pd.isna(value) and not low<=float(value)<=high:
            issue('outside_training_range','warning',col,f'{float(value):g} outside training range [{low:g}, {high:g}].')
    available=pd.Timestamp(res.scoring['model_available_date'])
    if row.engage_date<available:
        issue('retrospective_snapshot','warning','engage_date',f'Engaged {row.engage_date.date()}, before the model existed ({available.date()}).')
    days=(res.snapshot_date-row.engage_date).days
    if pool=='open' and days>res.stale_days:
        issue('stale_open_record','warning','engage_date',f'Open {days} days at the snapshot; {policy.STALE_QUANTILE:.0%} of training deals closed within {res.stale_days:.0f} days.')
    data={'issues':issues,'blocking':[i['code'] for i in issues if i['severity']=='blocking'],
          'warnings':[i['code'] for i in issues if i['severity']=='warning'],'imputed_inputs':imputed}
    return ToolResult('ok',data,[source('data/crm (joined pipeline row)',opportunity_id,res.scoring['features']),
                                 source('models/model_card.json','training_categories,training_numeric_range',[])])


# ------------------------------------------------------------------------------- compute tools

def predict_win_probability(res,opportunity_id,pool='open'):
    X=res.model_inputs(opportunity_id,pool)
    p=float(res.scoring['model'].predict_proba(X)[0,1])
    if not 0<=p<=1 or math.isnan(p): return ToolResult('error',error='model returned an invalid probability')
    t=float(res.scoring['decision_threshold'])
    data={'win_probability':round(p,10),'loss_probability':round(1-p,10),'decision_threshold':round(t,10),
          'predicted_outcome':'Won' if p>=t else 'Lost','distance_to_threshold':round(p-t,10),
          'model':res.scoring['model'].name,'feature_set':res.scoring['feature_set'],
          'bundle_sha256':res.bundle_sha256,'calibration':'validation-fitted sigmoid; frozen base model'}
    return ToolResult('ok',data,[source('models/model_bundle.joblib',res.bundle_sha256[:12],res.scoring['features'])])


def sensitivities(model,X,reference):
    """src.explain.reference.reference_sensitivities for one row, batched into a single model call."""
    rows=[X]
    for col in X: rows.append(X.assign(**{col:reference[col]}))
    p=model.predict_proba(pd.concat(rows,ignore_index=True))[:,1]
    return pd.Series(p[0]-p[1:],index=X.columns)


def explain_prediction(res,opportunity_id,pool='open'):
    X=res.model_inputs(opportunity_id,pool)
    effect=sensitivities(res.scoring['model'],X,res.scoring['reference'])
    effect=effect[effect.abs()>1e-10]
    ordered=sorted(effect.items(),key=lambda kv:(-abs(kv[1]),kv[0]))
    factors=[{'feature':k,'value':jsonable(X.iloc[0][k]),'reference':jsonable(res.scoring['reference'][k]),
              'delta_p_win':round(float(v),10),'direction':'supports win' if v>0 else 'opposes win'} for k,v in ordered[:5]]
    data={'factors':factors,'method':'single-feature reference sensitivity of the selected calibrated model',
          'interpretation':'predictive association inside this model; not additive and not a causal business effect'}
    return ToolResult('ok',data,[source('models/explanation_reference.json','training medians/modes',list(effect.index))])


def check_model_agreement(res,opportunity_id,pool='open'):
    """Committee check: each learned model's score as a percentile of its own validation scores."""
    X=res.model_inputs(opportunity_id,pool)
    selected=res.trained.selected; members={}
    for name,quantiles in res.card['committee_validation_quantiles'].items():
        p=float(res.trained.models[name].predict_proba(X)[0,1])
        members[name]={'raw_win_probability':round(p,10),
                       'validation_percentile':round(float(np.interp(p,quantiles,np.linspace(0,1,101))),10)}
    side=lambda v:v['validation_percentile']>=.5
    reference=side(members[selected])
    others=[n for n in members if n!=selected]
    disagree=[n for n in others if side(members[n])!=reference]
    spread=max(m['validation_percentile'] for m in members.values())-min(m['validation_percentile'] for m in members.values())
    data={'members':members,'selected_model':selected,'disagreeing_models':disagree,
          'disagreement_share':round(len(disagree)/len(others),10),'percentile_spread':round(spread,10),
          'committee_split':len(disagree)/len(others)>=policy.COMMITTEE_SPLIT,
          'note':'Percentiles use each model\'s validation score distribution (models/model_card.json); no test row is used.'}
    return ToolResult('ok',data,[source('models/trained_models.joblib',res.trained.feature_set,list(members))])


def evaluate_evidence(res,beliefs):
    """Turn observations into an auditable assessment: evidence quality, conflicts, reliability, risk signal."""
    from src.agent.planner import assess
    return ToolResult('ok',assess(beliefs,res.card),[source('models/model_card.json','validation_roc_auc,validation_calibrated_roc_auc_ci95',[])])


# ---------------------------------------------------------------------------- local-write tools

def create_review_task(res,guard,outbox,task):
    """Write one review task as a local JSON file. Nothing is sent and no CRM record is changed."""
    path=guard.local_path(Path(outbox)/f"{task['task_id']}.json")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(task,indent=2,ensure_ascii=False),encoding='utf-8')
    return ToolResult('ok',{'task_id':task['task_id'],'path':str(path),'external_effect':False},
                      [source('local file',str(path.name),list(task))])


def export_agent_report(res,guard,folder,stem,trace,markdown):
    folder=guard.local_path(folder); folder.mkdir(parents=True,exist_ok=True)
    json_path=folder/f'{stem}.trace.json'; md_path=folder/f'{stem}.report.md'
    json_path.write_text(json.dumps(trace,indent=2,ensure_ascii=False),encoding='utf-8')
    md_path.write_text(markdown,encoding='utf-8')
    return ToolResult('ok',{'trace':str(json_path),'report':str(md_path)},[source('local file',json_path.name,[])])


# ------------------------------------------------------------- external tools: never executed

def _external(name):
    def refuse(*args,**kwargs):
        raise policy.PolicyViolation(f'{name} is an external action; it has no implementation and needs human approval')
    return refuse


TOOLS={t.name:t for t in [
    Tool('get_opportunity','read','opportunity','Pipeline fields of one dated Engaging (or outcome-masked) deal',get_opportunity),
    Tool('get_account_information','read','account','Account profile from accounts.csv',get_account_information),
    Tool('get_prior_sales_history','read','history','Strictly prior agent/account/product outcomes from the frozen archive',get_prior_sales_history),
    Tool('check_data_quality','compute','data_quality','Missing, imputed, unseen or out-of-range inputs; timing and staleness',check_data_quality),
    Tool('predict_win_probability','compute','prediction','Calibrated P(Won) from the frozen selected model',predict_win_probability),
    Tool('check_model_agreement','compute','agreement','Committee of learned models on validation percentile scales',check_model_agreement),
    Tool('explain_prediction','compute','explanation','Reference sensitivities of the selected calibrated model',explain_prediction),
    Tool('evaluate_evidence','compute','assessment','Evidence quality, conflicts, model reliability and risk signal',evaluate_evidence),
    Tool('create_review_task','local_write','','Write a human review task to the local outbox',create_review_task),
    Tool('export_agent_report','local_write','','Write the decision trace and report',export_agent_report),
    Tool('send_customer_email','communication','','Email a customer (never executed; human approval required)',_external('send_customer_email')),
    Tool('update_crm_record','external_write','','Change a CRM record (never executed; human approval required)',_external('update_crm_record')),
    Tool('contact_third_party','communication','','Contact a third party (never executed; human approval required)',_external('contact_third_party')),
]}
INFORMATION_TOOLS=['get_opportunity','get_account_information','check_data_quality','get_prior_sales_history',
                   'predict_win_probability','check_model_agreement','explain_prediction']


def call(res,name,*args,**kwargs):
    """Run one tool with fault injection and timing. Exceptions become error results, never fake data."""
    start=perf_counter()
    try:
        if name in res.faults: raise RuntimeError(f'{res.faults[name]} (injected fault for evaluation)')
        result=TOOLS[name].fn(res,*args,**kwargs)
    except policy.PolicyViolation: raise
    except Exception as exc:
        result=ToolResult('error',error=f'{type(exc).__name__}: {exc}')
    return result,round((perf_counter()-start)*1000,3)
