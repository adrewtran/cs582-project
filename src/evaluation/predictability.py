"""Why is real-data prediction weak? A predictability study that never selects anything on the test period.

    python -m src.evaluation.predictability [--output RUN] [--quick]

Pre-registered in docs/PREREGISTRATION.md (section G1). Development data are the deals whose outcome was known
before the test period starts (training, validation and the purged rows that closed before test_start). Every
analysis and every choice below uses development data only. The test period is touched once, at the end, after
the chosen candidate is written to analysis/predictability/selection_lock.json.

Parts:
 1. outcome base rates and drift over time;            5. whole-model label-permutation tests;
 2. missingness and support of the open pipeline;      6. rolling-origin learnability study of 9 candidates;
 3. deal age: the outcome depends on time to close;    7. a time-aware (landmark) reformulation;
 4. univariate signal and stability of each feature;   8. statistical power; 9. single locked test confirmation.
"""
import argparse
from dataclasses import replace
from datetime import datetime,timezone
import json
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency,mannwhitneyu,norm,spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from src.data.crm import CATEGORICAL,NUMERIC,build
from src.data.history import HISTORY_COLUMNS,build_history
from src.data.preprocess import make_preprocessor
from src.data.split import asof_split
from src.models.zoo import SEED,NativePreprocessor,fit_model
from src.outputs import DEFAULT_OUTPUT,Outputs,write_json

AGES=[0,7,14,30,60,90,120]                  # landmark ages in days since engagement
FOLDS=4                                     # rolling-origin folds inside the development period
CALENDAR=['engage_year','engage_month','engage_quarter','engage_dayofweek']
WIN_RATES=['agent_win_rate','account_win_rate','product_win_rate']
INCUMBENT='lr_history'                      # the configuration src.train selected on validation


def wilson(k,n,z=1.96):
    if n==0: return (np.nan,np.nan)
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))
    return ((c-h)/d,(c+h)/d)


def boot_auc(y,p,draws,seed=SEED):
    y=np.asarray(y); p=np.asarray(p); rng=np.random.default_rng(seed); v=[]
    for _ in range(draws):
        i=rng.integers(0,len(y),len(y))
        if 0<y[i].mean()<1: v.append(roc_auc_score(y[i],p[i]))
    return float(np.percentile(v,2.5)),float(np.percentile(v,97.5))


def hanley_se(auc,n_pos,n_neg):
    q1=auc/(2-auc); q2=2*auc*auc/(1+auc)
    return float(np.sqrt((auc*(1-auc)+(n_pos-1)*(q1-auc*auc)+(n_neg-1)*(q2-auc*auc))/(n_pos*n_neg)))


# ------------------------------------------------------------------------------------- data

class Study:
    def __init__(self,quick=False):
        self.quick=quick
        self.dataset=build(); self.split=asof_split(self.dataset)
        f=self.dataset.frame; s=self.split
        self.y=self.dataset.labels()
        self.role=s.manifest.role
        self.dev=f.index[(f.close_date<s.test_start)].to_numpy()              # outcome known before the test period
        self.cycle=(f.close_date-f.engage_date).dt.days
        self.open=self.dataset.extra['scorable_open_deals']
        self.snapshot=max(f.close_date.max(),f.engage_date.max(),self.open.engage_date.max())

    def with_history(self,rows,archive_rows):
        """Rows of the frame plus strictly prior history built from a frozen archive (archive_rows)."""
        f=self.dataset.frame
        h,_=build_history(f.loc[rows,['opportunity_id','engage_date','sales_agent','account','product']],f.loc[archive_rows])
        return f.loc[rows].join(h)


# ------------------------------------------------------------------------------ 1-3 descriptive

def base_rates(st):
    f=st.dataset.frame; rows=[]
    for role,g in f.groupby(st.role):
        k=int(g.is_won.sum()); lo,hi=wilson(k,len(g))
        rows.append({'role':role,'rows':len(g),'win_rate':k/len(g),'ci_low':lo,'ci_high':hi,
                     'used_for':'description only' if role=='test' else 'development'})
    dev=f.loc[st.dev]; month=dev.engage_date.dt.to_period('M').astype(str)
    monthly=dev.groupby(month).is_won.agg(['size','sum','mean']).rename(columns={'size':'deals','sum':'won','mean':'win_rate'})
    monthly[['ci_low','ci_high']]=[wilson(r.won,r.deals) for r in monthly.itertuples()]
    table=pd.crosstab(month,dev.is_won); chi=chi2_contingency(table)
    return pd.DataFrame(rows),monthly.reset_index(names='engage_month'),{'months':len(table),'chi2':float(chi.statistic),'dof':int(chi.dof),'p_value':float(chi.pvalue)}


def support(st):
    f=st.dataset.frame; o=st.open; prospect=st.dataset.extra['open_deals'].loc[lambda d:d.deal_stage.eq('Prospecting')]
    age=(st.snapshot-o.engage_date).dt.days; longest=int(st.cycle.max())
    cols=['account','sector','revenue','employees','year_established','office_location']
    rows=[{'population':name,'rows':len(g),**{f'missing_{c}':float(g[c].isna().mean()) for c in cols}}
          for name,g in [('closed (labelled)',f),('open Engaging (scored)',o),('open Prospecting (not scored)',prospect)]]
    facts={'longest_closed_cycle_days':longest,'median_closed_cycle_days':float(st.cycle.median()),
           'open_age_median_days':float(age.median()),'open_older_than_any_closed_cycle':float((age>longest).mean()),
           'open_older_than_any_closed_cycle_count':int((age>longest).sum()),
           'open_missing_account':float(o.account.isna().mean()),'closed_missing_account':float(f.account.isna().mean()),
           'snapshot_date':str(st.snapshot.date())}
    return pd.DataFrame(rows),facts


def deal_age(st):
    """Outcome vs time to close, on development rows only; and the landmark view P(won | still open at age a)."""
    f=st.dataset.frame.loc[st.dev]; c=st.cycle.loc[st.dev]
    won,lost=c[f.is_won.eq(1)],c[f.is_won.eq(0)]
    test=mannwhitneyu(won,lost)
    rows=[]
    for a in AGES:
        g=f.loc[c>a]; k=int(g.is_won.sum()); lo,hi=wilson(k,len(g))
        rows.append({'age_days':a,'still_open':len(g),'share_of_deals':len(g)/len(f),'win_rate_if_still_open':k/len(g),'ci_low':lo,'ci_high':hi})
    purged=st.dataset.frame.loc[st.role.eq('purged_train')]
    facts={'median_cycle_won':float(won.median()),'median_cycle_lost':float(lost.median()),
           'mann_whitney_p':float(test.pvalue),'closed_within_14_days_win_rate':float(f.loc[c<=14].is_won.mean()),
           'closed_after_14_days_win_rate':float(f.loc[c>14].is_won.mean()),
           'purged_train_rows':len(purged),'purged_train_win_rate':float(purged.is_won.mean()),
           'train_win_rate':float(st.y.loc[st.split.train].mean()),
           'note':'time to close is known only after the outcome, so it is never a feature at engagement'}
    return pd.DataFrame(rows),facts


# --------------------------------------------------------------------------- 4 per-feature signal

def univariate(st,draws,perms):
    """One feature at a time: fit on training rows, score validation rows (development data only)."""
    s=st.split; frame=st.with_history(np.concatenate([s.train,s.validation]),s.train)
    yt,yv=st.y.loc[s.train].to_numpy(),st.y.loc[s.validation].to_numpy()
    rng=np.random.default_rng(SEED); rows=[]
    for col in CATEGORICAL+NUMERIC+[c for c in HISTORY_COLUMNS if not c.startswith('global')]:
        cat=col in CATEGORICAL
        data=replace(st.dataset,frame=frame,categorical=[col] if cat else [],numeric=[] if cat else [col])
        prep=make_preprocessor(data,scale_numeric=True)
        xt=prep.fit_transform(frame.loc[s.train,[col]]); xv=prep.transform(frame.loc[s.validation,[col]])
        if np.nanstd(np.asarray(xt,dtype=float))==0: continue
        p=LogisticRegression(max_iter=2000).fit(xt,yt).predict_proba(xv)[:,1]
        auc=roc_auc_score(yv,p); lo,hi=boot_auc(yv,p,draws)
        null=np.array([roc_auc_score(rng.permutation(yv),p) for _ in range(perms)])
        rows.append({'feature':col,'kind':'categorical' if cat else ('history' if col in HISTORY_COLUMNS else 'numeric'),
                     'levels_or_unique':int(frame.loc[s.train,col].nunique()),'validation_auc':auc,'ci_low':lo,'ci_high':hi,
                     'perm_p':float((1+(np.abs(null-.5)>=abs(auc-.5)).sum())/(perms+1))})
    out=pd.DataFrame(rows).sort_values('validation_auc',ascending=False)
    # Benjamini-Hochberg across features (two-sided permutation p-values).
    p=out.perm_p.to_numpy(); order=np.argsort(p); ranked=p[order]*len(p)/np.arange(1,len(p)+1)
    q=np.empty_like(p); q[order]=np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    out['bh_q']=q
    return out


def stability(st,min_rows=20):
    """Do category win rates repeat from training to validation?  And is the spread larger than chance?"""
    f=st.dataset.frame; s=st.split; rows=[]
    for col in ['sales_agent','manager','regional_office','product','series','sector','office_location','account']:
        tr=f.loc[s.train].groupby(col).is_won.agg(['size','mean']); va=f.loc[s.validation].groupby(col).is_won.agg(['size','mean'])
        both=tr.join(va,lsuffix='_train',rsuffix='_val',how='inner').loc[lambda d:(d.size_train>=min_rows)&(d.size_val>=min_rows)]
        table=pd.crosstab(f.loc[s.train,col],f.loc[s.train,'is_won']); chi=chi2_contingency(table)
        rho=spearmanr(both.mean_train,both.mean_val) if len(both)>=4 else None
        rows.append({'feature':col,'levels_train':len(tr),'levels_compared':len(both),
                     'train_heterogeneity_chi2_p':float(chi.pvalue),
                     'spearman_train_vs_validation':float(rho.statistic) if rho else np.nan,
                     'spearman_p':float(rho.pvalue) if rho else np.nan})
    return pd.DataFrame(rows)


def drift(st):
    """Population stability index (training -> validation, training -> open pipeline) and adversarial AUC."""
    f=st.dataset.frame; s=st.split; o=st.open; rows=[]
    def psi(a,b,cat):
        if cat:
            pa=a.fillna('Missing').value_counts(normalize=True); pb=b.fillna('Missing').value_counts(normalize=True)
            keys=pa.index.union(pb.index); pa=pa.reindex(keys,fill_value=0); pb=pb.reindex(keys,fill_value=0)
        else:
            edges=np.unique(np.nanquantile(a.dropna(),np.linspace(0,1,11)))
            if len(edges)<3: return 0.
            cut=lambda v:pd.cut(v,np.r_[-np.inf,edges[1:-1],np.inf]).value_counts(normalize=True,sort=False)
            pa=cut(a.dropna()); pb=cut(b.dropna())
        pa,pb=np.clip(pa.to_numpy(),1e-4,None),np.clip(pb.to_numpy(),1e-4,None)
        return float(((pb-pa)*np.log(pb/pa)).sum())
    for col in CATEGORICAL+[c for c in NUMERIC if c not in CALENDAR]:
        cat=col in CATEGORICAL
        rows.append({'feature':col,'psi_train_to_validation':psi(f.loc[s.train,col],f.loc[s.validation,col],cat),
                     'psi_train_to_open_pipeline':psi(f.loc[s.train,col],o[col],cat)})
    # Adversarial validation without calendar columns: can a model tell training rows from validation rows?
    cols=[c for c in CATEGORICAL+NUMERIC if c not in CALENDAR]
    data=replace(st.dataset,categorical=CATEGORICAL,numeric=[c for c in NUMERIC if c not in CALENDAR])
    idx=np.concatenate([s.train,s.validation]); z=np.r_[np.zeros(len(s.train)),np.ones(len(s.validation))]
    oof=np.zeros(len(idx))
    for tr,te in StratifiedKFold(5,shuffle=True,random_state=SEED).split(idx,z):
        prep=make_preprocessor(data,scale_numeric=True)
        X=f.loc[idx,cols]
        oof[te]=LogisticRegression(max_iter=2000).fit(prep.fit_transform(X.iloc[tr]),z[tr]).predict_proba(prep.transform(X.iloc[te]))[:,1]
    return pd.DataFrame(rows),{'adversarial_auc_train_vs_validation':float(roc_auc_score(z,oof)),
                               'note':'5-fold logistic regression on non-calendar inputs; 0.5 = no detectable covariate shift'}


# ------------------------------------------------------------------------- 5 permutation tests

def permutation_tests(st,perms_lr,perms_cb):
    """Refit the production model on shuffled training labels; compare the observed validation AUC to that null."""
    s=st.split; hist=st.with_history(st.dataset.frame.index.to_numpy(),s.train)
    variants={'history/logistic_regression':(replace(st.dataset,frame=hist,numeric=NUMERIC+HISTORY_COLUMNS),'logistic_regression',perms_lr),
              'raw/catboost':(st.dataset,'catboost',perms_cb)}
    rng=np.random.default_rng(SEED); rows=[]; nulls={}
    for name,(data,model,perms) in variants.items():
        yv=data.labels().loc[s.validation]
        observed=roc_auc_score(yv,fit_model(model,data,s.train,s.validation,quick=st.quick).predict_proba(data.features().loc[s.validation])[:,1])
        null=[]
        for _ in range(perms):
            frame=data.frame.copy(); frame.loc[s.train,'is_won']=rng.permutation(frame.loc[s.train,'is_won'].to_numpy())
            fitted=fit_model(model,replace(data,frame=frame),s.train,s.validation,quick=st.quick)
            null.append(roc_auc_score(yv,fitted.predict_proba(data.features().loc[s.validation])[:,1]))
        null=np.array(null); nulls[name]=null
        rows.append({'configuration':name,'observed_validation_auc':observed,'permutations':perms,
                     'null_mean':float(null.mean()),'null_95th':float(np.percentile(null,95)),
                     'p_value':float((1+(null>=observed).sum())/(perms+1))})
    return pd.DataFrame(rows),nulls


# ----------------------------------------------------------------------- 6 rolling-origin study

CANDIDATES={
    'dummy':'class prior',
    'lr_raw':'logistic regression, 18 raw inputs, C=1 (src.train raw/LR)',
    'lr_history':'logistic regression, raw + strictly prior history, C=1 (the selected production configuration)',
    'lr_raw_strong':'logistic regression, raw inputs, strong L2 (C=0.05)',
    'lr_history_strong':'logistic regression, raw + history, strong L2 (C=0.05)',
    'lr_history_rates_only':'logistic regression on the three smoothed prior win rates only',
    'lr_agent_product':'logistic regression, raw + agent x product interaction, C=0.1',
    'catboost_shallow':'CatBoost depth 3, l2 10, 300 iterations, raw inputs',
    'catboost_history':'CatBoost depth 5, 200 iterations, raw + history',
}


def folds(st):
    """Expanding-window folds inside the development period; same purge rule as src.data.split."""
    f=st.dataset.frame.loc[st.dev]; dates=f.engage_date.sort_values(kind='stable')
    cuts=[dates.iloc[int(len(dates)*q)] for q in np.linspace(.4,1,FOLDS+1)[:-1]]+[st.split.test_start]
    for k in range(FOLDS):
        start,end=cuts[k],cuts[k+1]
        train=f.index[(f.engage_date<start)&(f.close_date<start)].to_numpy()
        test=f.index[(f.engage_date>=start)&(f.engage_date<end)&(f.close_date<end)].to_numpy()
        yield k,start,train,test


def design(st,name,frame,train,test):
    """Return (X_train, X_test, fit) for one candidate; preprocessing is fitted on the training rows only."""
    hist=name in ('lr_history','lr_history_strong','lr_history_rates_only','catboost_history')
    cat=[] if name=='lr_history_rates_only' else list(CATEGORICAL)
    num=WIN_RATES if name=='lr_history_rates_only' else NUMERIC+(HISTORY_COLUMNS if hist else [])
    frame=frame.copy()
    if name=='lr_agent_product': frame['agent_x_product']=frame.sales_agent.astype(str)+'|'+frame['product'].astype(str); cat.append('agent_x_product')
    data=replace(st.dataset,frame=frame,categorical=cat,numeric=num)
    if name.startswith('catboost'):
        from catboost import CatBoostClassifier
        depth,l2,it=(3,10,300) if name=='catboost_shallow' else (5,5,200)
        if st.quick: it=20
        prep=NativePreprocessor(cat,num).fit(frame.loc[train,cat+num])
        clf=CatBoostClassifier(iterations=it,depth=depth,l2_leaf_reg=l2,learning_rate=.03,random_seed=SEED,thread_count=2,
                               allow_writing_files=False,verbose=False).fit(prep.transform(frame.loc[train]),frame.loc[train,'is_won'],cat_features=cat)
        return clf.predict_proba(prep.transform(frame.loc[test]))[:,1]
    if name=='dummy': return np.full(len(test),frame.loc[train,'is_won'].mean())
    C={'lr_raw_strong':.05,'lr_history_strong':.05,'lr_agent_product':.1}.get(name,1.)
    prep=make_preprocessor(data,scale_numeric=True)
    xt=np.asarray(prep.fit_transform(frame.loc[train,cat+num]),dtype=float)
    clf=LogisticRegression(C=C,max_iter=3000,random_state=SEED).fit(xt,frame.loc[train,'is_won'])
    return clf.predict_proba(np.asarray(prep.transform(frame.loc[test,cat+num]),dtype=float))[:,1]


def learnability(st,draws):
    preds={}; rows=[]
    for k,start,train,test in folds(st):
        frame=st.with_history(np.concatenate([train,test]),train)        # the fold's training rows are its archive
        y=frame.loc[test,'is_won'].to_numpy()
        for name in CANDIDATES:
            p=design(st,name,frame,train,test); preds[(k,name)]=(y,p)
            rows.append({'fold':k,'fold_start':str(start.date()),'train_rows':len(train),'eval_rows':len(test),
                         'eval_win_rate':float(y.mean()),'candidate':name,'auc':roc_auc_score(y,p) if name!='dummy' else .5})
    per_fold=pd.DataFrame(rows)
    rng=np.random.default_rng(SEED); summary=[]
    samples={name:[] for name in CANDIDATES}; deltas={name:[] for name in CANDIDATES}
    for _ in range(draws):
        idx={k:rng.integers(0,len(preds[(k,'dummy')][0]),len(preds[(k,'dummy')][0])) for k in range(FOLDS)}
        mean=lambda name:np.mean([roc_auc_score(preds[(k,name)][0][idx[k]],preds[(k,name)][1][idx[k]]) if name!='dummy' else .5 for k in range(FOLDS)])
        base=mean(INCUMBENT)
        for name in CANDIDATES:
            v=mean(name); samples[name].append(v); deltas[name].append(v-base)
    for name,desc in CANDIDATES.items():
        g=per_fold.loc[per_fold.candidate.eq(name)]
        summary.append({'candidate':name,'description':desc,'mean_fold_auc':float(g.auc.mean()),'min_fold_auc':float(g.auc.min()),
                        'max_fold_auc':float(g.auc.max()),'ci_low':float(np.percentile(samples[name],2.5)),'ci_high':float(np.percentile(samples[name],97.5)),
                        'delta_vs_incumbent':float(g.auc.mean()-per_fold.loc[per_fold.candidate.eq(INCUMBENT)].auc.mean()),
                        'delta_ci_low':float(np.percentile(deltas[name],2.5)),'delta_ci_high':float(np.percentile(deltas[name],97.5))})
    return per_fold,pd.DataFrame(summary).sort_values('mean_fold_auc',ascending=False)


# --------------------------------------------------------------------- 7 landmark reformulation

def landmark_rows(st,rows,cutoff_end,window=None):
    """One row per (deal, age) while the deal is still open at that age; the label must be known by cutoff_end."""
    f=st.dataset.frame; out=[]
    for a in AGES:
        when=f.loc[rows,'engage_date']+pd.Timedelta(days=a)
        keep=rows[(st.cycle.loc[rows]>a).to_numpy()&(f.loc[rows,'close_date']<cutoff_end).to_numpy()
                  &((when>=window[0]).to_numpy() if window else True)&(when<(window[1] if window else cutoff_end)).to_numpy()]
        out.append(pd.DataFrame({'row':keep,'age':a}))
    return pd.concat(out,ignore_index=True)


def landmark(st,draws):
    """Score a deal at age a given that it is still open (survival 'landmarking'), on validation-period landmarks."""
    s=st.split; f=st.dataset.frame
    frame=st.with_history(st.dev,s.train)
    train=landmark_rows(st,st.dev[f.loc[st.dev,'engage_date']<s.validation_start],s.validation_start)
    val=landmark_rows(st,st.dev,s.test_start,window=(s.validation_start,s.test_start))
    def X(rows): return frame.loc[rows.row].reset_index(drop=True).assign(deal_age=rows.age.to_numpy())
    xt,xv=X(train),X(val); yt,yv=xt.is_won.to_numpy(),xv.is_won.to_numpy()
    results=[]
    for name,cat,num in [('age_only',[],['deal_age']),('features_only (engagement model)',CATEGORICAL,NUMERIC+HISTORY_COLUMNS),
                         ('features_plus_age',CATEGORICAL,NUMERIC+HISTORY_COLUMNS+['deal_age'])]:
        data=replace(st.dataset,frame=xt,categorical=cat,numeric=num); prep=make_preprocessor(data,scale_numeric=True)
        clf=LogisticRegression(max_iter=3000,random_state=SEED).fit(np.asarray(prep.fit_transform(xt[cat+num]),dtype=float),yt)
        p=clf.predict_proba(np.asarray(prep.transform(xv[cat+num]),dtype=float))[:,1]
        lo,hi=boot_auc(yv,p,draws)
        within=[roc_auc_score(yv[xv.deal_age.eq(a)],p[xv.deal_age.eq(a)]) for a in AGES if 0<yv[xv.deal_age.eq(a)].mean()<1 and xv.deal_age.eq(a).sum()>=30]
        results.append({'model':name,'train_landmarks':len(xt),'validation_landmarks':len(xv),'pooled_validation_auc':roc_auc_score(yv,p),
                        'ci_low':lo,'ci_high':hi,'mean_within_age_auc':float(np.mean(within)),
                        'note':'pooled AUC mixes ages; within-age AUC measures feature signal beyond deal age'})
    open_age=(st.snapshot-st.open.engage_date).dt.days
    return pd.DataFrame(results),{'open_deals_within_landmark_support':float((open_age<=max(AGES)).mean()),
                                  'max_landmark_age':max(AGES)}


# ------------------------------------------------------------------------------------ 8 power

def power(st):
    s=st.split; yv=st.y.loc[s.validation]; npos,nneg=int(yv.sum()),int(len(yv)-yv.sum()); rows=[]
    for auc in [.55,.57,.60,.65,.70]:
        se=hanley_se(auc,npos,nneg)
        # deals needed (same class mix) so that the 95% CI lower bound clears 0.5
        n=next(m for m in range(50,200001,25) if auc-1.96*hanley_se(auc,max(2,int(m*yv.mean())),max(2,int(m*(1-yv.mean()))))>.5)
        rows.append({'true_auc':auc,'validation_rows':len(yv),'se_at_validation_size':se,'ci_halfwidth':1.96*se,
                     'rows_needed_for_ci_above_0.5':n})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- 9 locked confirmation

def confirm(st,summary,folder):
    """Lock the decision before the test period is scored; score once; report both models side by side."""
    best=summary.iloc[0]
    adopt=best.candidate!=INCUMBENT and best.delta_ci_low>0
    choice=best.candidate if adopt else INCUMBENT
    lock={'rule':'highest mean rolling-origin AUC; adopt only if the paired 95% CI vs the incumbent excludes 0',
          'best_candidate':best.candidate,'best_delta':float(best.delta_vs_incumbent),'best_delta_ci':[float(best.delta_ci_low),float(best.delta_ci_high)],
          'adopted':bool(adopt),'confirmed_configuration':choice,'locked_utc':datetime.now(timezone.utc).isoformat(),
          'test_scored_before_lock':False}
    write_json(folder/'selection_lock.json',lock)
    s=st.split; rows=[]
    frame=st.with_history(st.dataset.frame.index.to_numpy(),s.train)
    for name in dict.fromkeys([INCUMBENT,best.candidate]):
        for part,idx in [('validation',s.validation),('test',s.test)]:
            p=design(st,name,frame,s.train,idx); y=frame.loc[idx,'is_won'].to_numpy(); lo,hi=boot_auc(y,p,400 if st.quick else 2000)
            rows.append({'candidate':name,'role':'incumbent' if name==INCUMBENT else 'best development candidate','part':part,'auc':roc_auc_score(y,p),'ci_low':lo,'ci_high':hi})
    return lock,pd.DataFrame(rows)


# --------------------------------------------------------------------------------- figure + main

def figure(uni,nulls,perm,ages,summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluation.figures import save
    ink='#0E3B43'; teal='#0F7A6C'; grey='#9AA7B2'
    fig,axes=plt.subplots(2,2,figsize=(13,8.6))
    u=uni.head(14).iloc[::-1]; ax=axes[0,0]
    ax.errorbar(u.validation_auc,range(len(u)),xerr=[u.validation_auc-u.ci_low,u.ci_high-u.validation_auc],fmt='o',color=teal,ecolor=grey,capsize=2)
    ax.axvline(.5,color=ink,lw=1,ls='--'); ax.set_yticks(range(len(u)),u.feature,fontsize=8)
    ax.set_title('(a) One feature at a time: validation ROC-AUC, 95% CI',fontsize=10,color=ink,fontweight='bold')
    ax=axes[0,1]; name='history/logistic_regression'; null=nulls[name]; obs=perm.set_index('configuration').loc[name]
    ax.hist(null,bins=25,color=grey); ax.axvline(obs.observed_validation_auc,color=teal,lw=2.5)
    ax.text(obs.observed_validation_auc,ax.get_ylim()[1]*.92,f"  observed {obs.observed_validation_auc:.3f}\n  p = {obs.p_value:.3f}",color=teal,fontsize=9,va='top')
    ax.set_title(f'(b) Label-permutation null ({int(obs.permutations)} refits): history/LR',fontsize=10,color=ink,fontweight='bold'); ax.set_xlabel('validation ROC-AUC')
    ax=axes[1,0]
    ax.errorbar(ages.age_days,ages.win_rate_if_still_open,yerr=[ages.win_rate_if_still_open-ages.ci_low,ages.ci_high-ages.win_rate_if_still_open],fmt='o-',color=teal,ecolor=grey,capsize=3)
    ax.set_title('(c) Win rate of deals still open at each age (development rows)',fontsize=10,color=ink,fontweight='bold')
    ax.set_xlabel('days since engagement'); ax.set_ylabel('win rate'); ax.set_ylim(.4,1)
    ax=axes[1,1]; sm=summary.iloc[::-1]
    ax.errorbar(sm.mean_fold_auc,range(len(sm)),xerr=[sm.mean_fold_auc-sm.ci_low,sm.ci_high-sm.mean_fold_auc],fmt='o',color=teal,ecolor=grey,capsize=2)
    ax.axvline(.5,color=ink,lw=1,ls='--'); ax.set_yticks(range(len(sm)),sm.candidate,fontsize=8)
    ax.set_title(f'(d) {FOLDS}-fold rolling-origin ROC-AUC (development period only)',fontsize=10,color=ink,fontweight='bold')
    for a in axes.ravel(): a.spines[['top','right']].set_visible(False)
    save(fig,out.figures/'predictability.png')


def run(output_dir=DEFAULT_OUTPUT,quick=False):
    start=perf_counter(); out=Outputs(output_dir); folder=out.analysis/'predictability'
    folder.mkdir(parents=True,exist_ok=True); out.figures.mkdir(parents=True,exist_ok=True)
    st=Study(quick); draws=200 if quick else 2000
    rates,monthly,chi=base_rates(st); rates.to_csv(folder/'base_rates.csv',index=False); monthly.to_csv(folder/'monthly_win_rate.csv',index=False)
    miss,support_facts=support(st); miss.to_csv(folder/'missingness_and_support.csv',index=False)
    ages,age_facts=deal_age(st); ages.to_csv(folder/'deal_age_landmarks.csv',index=False)
    print('PREDICTABILITY: univariate signal...',flush=True)
    uni=univariate(st,draws,200 if quick else 2000); uni.to_csv(folder/'univariate_signal.csv',index=False)
    stab=stability(st); stab.to_csv(folder/'category_stability.csv',index=False)
    psi,adv=drift(st); psi.to_csv(folder/'drift_psi.csv',index=False)
    print('PREDICTABILITY: permutation tests...',flush=True)
    perm,nulls=permutation_tests(st,20 if quick else 200,5 if quick else 50); perm.to_csv(folder/'permutation_tests.csv',index=False)
    print('PREDICTABILITY: rolling-origin learnability...',flush=True)
    per_fold,summary=learnability(st,100 if quick else 1000)
    per_fold.to_csv(folder/'learnability_folds.csv',index=False); summary.to_csv(folder/'learnability_summary.csv',index=False)
    lm,lm_facts=landmark(st,draws); lm.to_csv(folder/'landmark_models.csv',index=False)
    pw=power(st); pw.to_csv(folder/'power.csv',index=False)
    lock,confirmation=confirm(st,summary,folder); confirmation.to_csv(folder/'confirmation.csv',index=False)
    figure(uni,nulls,perm,ages,summary,out)
    significant=uni.loc[uni.bh_q<.05,'feature'].tolist()
    result={'status':'complete','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full','elapsed_seconds':perf_counter()-start,
            'development_rows':len(st.dev),'month_heterogeneity':chi,'support':support_facts,'deal_age':age_facts,
            'features_significant_after_bh':significant,'adversarial':adv,
            'permutation':perm.to_dict('records'),'learnability_best':summary.iloc[0].to_dict(),
            'landmark':lm_facts,'lock':lock,
            'confirmation':confirmation.to_dict('records'),
            'protocol':'docs/PREREGISTRATION.md G1; development data only until selection_lock.json is written'}
    write_json(folder/'summary.json',result)
    print(f"PREDICTABILITY COMPLETE in {result['elapsed_seconds']:.0f}s: best development candidate {lock['best_candidate']}, adopted={lock['adopted']}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=lambda p:__import__('pathlib').Path(p),default=DEFAULT_OUTPUT)
    parser.add_argument('--quick',action='store_true',help='Few permutations and draws; not reportable')
    args=parser.parse_args(); run(args.output,quick=args.quick)


if __name__=='__main__': main()
