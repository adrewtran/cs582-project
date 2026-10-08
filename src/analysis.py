"""Reproducible diagnostic figures. Never used for model/threshold selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score,roc_curve,ConfusionMatrixDisplay,precision_recall_curve,average_precision_score
from sklearn.tree import DecisionTreeClassifier
from src.evaluation import metrics


def save(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def eda(dataset,split,out):
    figs=out/'figures'; figs.mkdir(parents=True,exist_ok=True)
    train=dataset.frame.loc[split.train]
    fig,ax=plt.subplots(figsize=(6,4))
    train.deal_stage.value_counts().reindex(['Lost','Won']).plot.bar(ax=ax,color=['#ea9768','#278b9c'],rot=0)
    ax.set(title='Training outcomes (labels available at cutoff)',ylabel='Deals')
    save(fig,figs/'eda_train_outcomes.png')
    fig,ax=plt.subplots(figsize=(7,4))
    train.groupby(train.engage_date.dt.to_period('M')).size().plot.bar(ax=ax,color='#278b9c')
    ax.set(title='Training engagements by month',ylabel='Deals',xlabel='Engagement month')
    save(fig,figs/'eda_train_months.png')
    products=train.groupby('product').is_won.agg(['count','mean']).sort_values('mean')
    products.to_csv(out/'eda_product_training.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    products['mean'].plot.barh(ax=ax,color='#278b9c')
    ax.set(title='Training win rate by product (descriptive only)',xlabel='Win rate',xlim=(0,1))
    save(fig,figs/'eda_product_training.png')
    closed=dataset.features().isna().mean()
    opened=dataset.extra['scorable_open_deals'][dataset.feature_columns].isna().mean()
    missing=pd.DataFrame({'closed_missing_fraction':closed,'engaging_missing_fraction':opened})
    missing.to_csv(out/'missingness.csv')
    shown=missing.loc[missing.max(axis=1)>0]
    if not len(shown): shown=missing.head(1)
    fig,ax=plt.subplots(figsize=(8,4))
    shown.plot.barh(ax=ax,color=['#278b9c','#ea9768'])
    ax.set(title='Input coverage shifts in open deals',xlabel='Missing fraction',xlim=(0,1))
    ax.legend(['Closed','Engaging'],loc='lower right',fontsize=8)
    save(fig,figs/'eda_missingness.png')


def evaluation_figures(y,probabilities,thresholds,calibrated,out):
    fig,ax=plt.subplots(figsize=(7,5))
    for name,p in probabilities.items():
        fpr,tpr,_=roc_curve(y,p)
        ax.plot(fpr,tpr,label=f'{name}: {roc_auc_score(y,p):.3f}')
        cmfig,cmax=plt.subplots(figsize=(4.5,4))
        ConfusionMatrixDisplay.from_predictions(y,p>=thresholds[name],labels=[0,1],display_labels=['Lost','Won'],ax=cmax,colorbar=False)
        cmax.set_title(name+' (validation threshold)',fontsize=10)
        save(cmfig,out/'figures'/f'confusion_{name}.png')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(title='Exploratory held-out ROC curves',xlabel='False positive rate',ylabel='True positive rate')
    ax.legend(fontsize=8); save(fig,out/'figures'/'roc_curves.png')
    fig,ax=plt.subplots(figsize=(7,5))
    for name,p in probabilities.items():
        precision,recall,_=precision_recall_curve(y,p)
        ax.plot(recall,precision,label=f'{name}: AP {average_precision_score(y,p):.3f}')
    ax.axhline(float(np.mean(y)),ls='--',color='gray',label='Test prevalence')
    ax.set(title='Exploratory held-out precision–recall',xlabel='Recall (Won)',ylabel='Precision (Won)',xlim=(0,1),ylim=(0,1.02))
    ax.legend(fontsize=8); save(fig,out/'figures'/'precision_recall_curves.png')
    fig,ax=plt.subplots(figsize=(6,4))
    for label,p in calibrated.items():
        actual,pred=calibration_curve(y,p,n_bins=6,strategy='quantile')
        ax.plot(pred,actual,'o-',label=label)
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(title='Reliability on test (quantile bins)',xlabel='Mean predicted win probability',ylabel='Observed win fraction',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8); save(fig,out/'figures'/'reliability.png')


def learning_curves(histories,out):
    for name in ['mlp','tabnet']:
        record=histories[name]; history=record['history']
        epochs=np.arange(1,len(history['loss'])+1)
        fig,(loss_ax,auc_ax)=plt.subplots(1,2,figsize=(9,3.5))
        loss_ax.plot(epochs,history['loss'],color='#278b9c')
        loss_ax.set(xlabel='Epoch',ylabel='Training loss')
        auc_ax.plot(epochs,history['validation_auc'],color='#278b9c')
        auc_ax.axvline(record['settings']['best_epoch'],ls='--',color='#db6849',label='Retained epoch')
        auc_ax.set(xlabel='Epoch',ylabel='Validation ROC-AUC'); auc_ax.legend(fontsize=8)
        fig.suptitle(name+' learning history (validation reused for selection)')
        save(fig,out/'figures'/f'learning_curve_{name}.png')


def native_importances(models,out):
    for name in ['logistic_regression','random_forest','tabnet','catboost']:
        model=models[name]
        if name=='logistic_regression':
            values=model.clf.coef_[0]
            description='Signed log-odds coefficient; numerics standardized'
        else:
            values=model.clf.feature_importances_
            description='Impurity importance' if name=='random_forest' else 'TabNet attention importance'
            if name=='catboost': description='CatBoost PredictionValuesChange (not causal)'
        table=pd.DataFrame({'feature':model.prep.get_feature_names_out(),'importance':values})
        table['absolute_importance']=table.importance.abs()
        table=table.sort_values('absolute_importance',ascending=False)
        table.to_csv(out/f'feature_importance_{name}.csv',index=False)
        shown=table.head(12).iloc[::-1]
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(shown.feature,shown.importance,color='#278b9c')
        ax.set(title=name,xlabel=description); save(fig,out/'figures'/f'importance_{name}.png')


def permutation_importance(model,X,y,out,quick=False):
    rng=np.random.default_rng(42); base=roc_auc_score(y,model.predict_proba(X)[:,1]); rows=[]
    for col in X:
        drops=[]
        for _ in range(1 if quick else 3):
            changed=X.copy(); changed[col]=rng.permutation(changed[col].to_numpy())
            drops.append(base-roc_auc_score(y,model.predict_proba(changed)[:,1]))
        rows.append({'feature':col,'mean_auc_drop':float(np.mean(drops)),'std_auc_drop':float(np.std(drops))})
    result=pd.DataFrame(rows).sort_values('mean_auc_drop',ascending=False)
    result.to_csv(out/'permutation_importance_test.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5)); shown=result.head(12).iloc[::-1]
    ax.barh(shown.feature,shown.mean_auc_drop,xerr=shown.std_auc_drop,color='#278b9c')
    ax.set(title='Held-out permutation sensitivity',xlabel='ROC-AUC drop; negative values are possible')
    save(fig,out/'figures'/'permutation_importance.png')


def leakage_control(dataset,split,honest_test_row,out):
    frame=dataset.frame; y=dataset.labels()
    model=DecisionTreeClassifier(max_depth=1,random_state=42)
    model.fit(frame.loc[split.train,['close_value']],y.loc[split.train])
    p=model.predict_proba(frame.loc[split.test,['close_value']])[:,1]
    leaky={'model':'LEAKY_close_value_stump',**metrics(y.loc[split.test],p),'purpose':'Invalid future information; illustration only'}
    honest=dict(honest_test_row,purpose='Pre-close feature set; no outcome predictors')
    pd.DataFrame([honest,leaky]).to_csv(out/'leakage_audit.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(['Honest selected model','LEAKY close_value'],[honest['roc_auc'],leaky['roc_auc']],color=['#278b9c','#db6849'])
    ax.set(ylim=(0,1.08),ylabel='Test ROC-AUC',title='Why outcome fields must be excluded')
    save(fig,out/'figures'/'leakage_audit.png')


def shap_audit(rf,X,ids,out,quick=False):
    # Explain the raw RF explicitly, not the selected/calibrated model.
    import shap
    sample=X.iloc[:(12 if quick else 64)]
    transformed=np.asarray(rf.prep.transform(sample),dtype=np.float32)
    explainer=shap.TreeExplainer(rf.clf,feature_perturbation='tree_path_dependent',model_output='raw')
    values=explainer.shap_values(transformed,check_additivity=True)
    values=np.asarray(values)[:,:,1]
    expected=float(np.asarray(explainer.expected_value)[1])
    prediction=rf.clf.predict_proba(transformed)[:,1]
    error=float(np.max(np.abs(expected+values.sum(axis=1)-prediction)))
    if error>1e-5: raise AssertionError('RF SHAP additivity check failed')
    names=rf.prep.get_feature_names_out()
    global_table=pd.DataFrame({'feature':names,'mean_absolute_shap':np.abs(values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False)
    global_table.to_csv(out/'rf_shap_global.csv',index=False)
    local=pd.DataFrame(values,columns=names); local.insert(0,'opportunity_id',np.asarray(ids)[:len(sample)])
    local.to_csv(out/'rf_shap_values.csv',index=False)
    audit={'model':'random_forest_raw','sample':'first dated test rows','rows':len(sample),
           'expected_probability':expected,'max_additivity_error':error,'method':'TreeSHAP tree_path_dependent',
           'warning':'Associations, not causal effects; not the calibrated scoring model'}
    (out/'shap_audit.json').write_text(json.dumps(audit,indent=2))
    fig,ax=plt.subplots(figsize=(9,5)); shown=global_table.head(12).iloc[::-1]
    ax.barh(shown.feature,shown.mean_absolute_shap,color='#278b9c')
    ax.set(title='Random Forest: TreeSHAP on test sample',xlabel='Mean absolute SHAP contribution to P(Won)')
    save(fig,out/'figures'/'rf_shap.png')
    local0=pd.Series(values[0],index=names); keep=local0.abs().nlargest(10).index
    shown=local0.loc[keep].sort_values()
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(shown.index,shown,color=np.where(shown>=0,'#278b9c','#db6849'))
    ax.set(title=f'RF explanation: {np.asarray(ids)[0]} (test example)',xlabel='SHAP contribution; association, not cause')
    save(fig,out/'figures'/'rf_shap_local.png')


def priority_checks(y,p,yval,pval,out):
    def groups(y,p,low,high):
        frame=pd.DataFrame({'observed':np.asarray(y),'probability':p})
        frame['priority']=np.where(p>=high,'High',np.where(p>=low,'Medium','Low'))
        result=frame.groupby('priority').agg(count=('observed','size'),observed_win_rate=('observed','mean'),mean_probability=('probability','mean')).reindex(['Low','Medium','High'])
        result['count']=result['count'].fillna(0).astype(int)
        return result.reset_index()
    groups(y,p,.4,.7).to_csv(out/'priority_test.csv',index=False)
    sensitivity=[]
    for low,high in [(.3,.6),(.4,.7),(.5,.8)]:
        table=groups(yval,pval,low,high); table['medium_min']=low; table['high_min']=high
        sensitivity.append(table)
    pd.concat(sensitivity).to_csv(out/'priority_validation_sensitivity.csv',index=False)
