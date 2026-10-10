"""Feedback learning: offline contextual-bandit policy improvement, demonstrated in a SYNTHETIC environment.

    python -m src.agent.learning [--output RUN] [--quick]

Pre-registered in docs/PREREGISTRATION.md (section G2).

Why synthetic. The CRM data hold no record of any action taken on a deal, of who reviewed it, or of whether a
review helped. Won/Lost is the outcome of the deal, not feedback on an action the agent took, so it is not used as
a reward. Real learning needs the reviewer feedback that src.agent.feedback collects during a pilot.

What is real and what is simulated.
  * contexts are real: the agent's evidence on real deals (validation and test replay with outcomes masked, and
    open deals without a blocking issue), summarized by feedback.context_from;
  * rewards are simulated: a reviewer usefulness score 1..5 drawn from an explicit reward model below, then mapped
    to [-1, 1] by feedback.reward, exactly as a human form answer would be.

The learning problem is a contextual bandit with logged feedback: one decision per deal, the reward is observed
only for the action taken, and the logging policy records its propensities. The agent learns which of
{ESCALATE_AT_RISK_REVIEW, REVIEW_UNCERTAIN, MONITOR_NO_TASK} reviewers find useful for a context. Blocked records
always go to data completion and external actions are never available: those are fixed safety rules.

  logging   epsilon-greedy around the fixed agent policy (epsilon 0.2), propensities stored with each record
  learner   per-action ridge regression of the reward on context features (direct method), greedy policy
  gate      safe policy improvement: fit on one half of the feedback, estimate (learned - fixed) on the other half
            with the doubly robust estimator; adopt only if the bootstrap 95% lower bound is above 0
  evaluate  exact expected reward of each policy on held-out deals (known only because the environment is
            simulated), against the fixed policy and an oracle

Environments: E0 no preference signal (control: nothing to learn); E1 reviewers agree with the fixed policy
(control: nothing to gain); E2 reviewers' preferences differ from the fixed rule in stated ways. This is not
reinforcement learning (no state transitions) and nothing here is a claim about real sales outcomes.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.agent import policy
from src.agent.feedback import FeedbackRecord,FeedbackStore,DEFAULT_STORE,context_from,reward
from src.outputs import DEFAULT_OUTPUT,Outputs,write_json

ACTIONS=['ESCALATE_AT_RISK_REVIEW','REVIEW_UNCERTAIN','MONITOR_NO_TASK']
FEATURES=['bias','distance_to_threshold','abs_distance','borderline','evidence_quality','conflict','disagreement_share','stale','out_of_range']
EPSILON=.2
RIDGE=1.0
SIGMA=.5                                       # reviewer noise on the continuous usefulness scale
LOG_SIZES=[100,250,500,1000,2000]
SEEDS=list(range(20))
CUTS=np.array([-.75,-.25,.25,.75])             # usefulness 1..5 <-> reward -1, -.5, 0, .5, 1


def phi(c):
    d=c['distance_to_threshold']
    return np.array([1.,d,abs(d),c['borderline'],c['evidence_quality'],c['conflict'],c['disagreement_share'],c['stale'],c['out_of_range']])


# ------------------------------------------------------------------------------- environments

def expected_reward(mu):
    """E[(u-3)/2] when u = clip(round(3 + 2 (mu + noise)), 1, 5); exact under Gaussian noise."""
    mu=np.asarray(mu,dtype=float)[...,None]
    cdf=norm.cdf((CUTS-mu)/SIGMA)
    probs=np.diff(np.concatenate([np.zeros_like(mu),cdf,np.ones_like(mu)],axis=-1),axis=-1)
    return (probs*np.array([-1,-.5,0,.5,1])).sum(-1)


def sample_usefulness(mu,rng):
    r=mu+rng.normal(0,SIGMA,np.shape(mu))
    return np.clip(np.round(3+2*r),1,5).astype(int)


def environment(name,X,fixed):
    """Mean continuous usefulness mu[n, action] for each context. All parameters are stated here."""
    n=len(X); a=np.array([ACTIONS.index(f) for f in fixed])
    if name=='E0_no_signal': return np.zeros((n,3))
    if name=='E1_agrees_with_fixed':
        mu=np.full((n,3),-.25); mu[np.arange(n),a]=.5; return mu
    if name=='E2_shifted_preferences':
        d,ev,conflict,share,stale=X[:,1],X[:,4],X[:,5],X[:,6],X[:,7]
        q=1/(1+np.exp(-(-.3-8*d+1.0*conflict+1.2*stale+.8*share)))     # simulated value of human attention
        esc=1.3*q-.5
        rev=.7*q-.1-.4*(ev<.5)                                          # thin-evidence reviews waste time
        mon=.3-1.0*q
        return np.column_stack([esc,rev,mon])
    raise ValueError(name)


ENVIRONMENTS={'E0_no_signal':'all actions equally useful on average: there is nothing to learn (false-adoption control)',
              'E1_agrees_with_fixed':'reviewers prefer exactly what the fixed policy does: learning cannot gain (control)',
              'E2_shifted_preferences':'reviewer value rises when the score is below the threshold, with conflicts, stale records or a split committee; '
                                       'reviews of thin-evidence deals waste time; the fixed rule does not encode this'}


# --------------------------------------------------------------------------- logging + learning

def log_feedback(X,fixed,mu,n,rng):
    """Epsilon-greedy logging around the fixed policy; returns indices, actions, propensities, usefulness."""
    idx=rng.integers(0,len(X),n)
    explore=rng.random(n)<EPSILON
    f=np.array([ACTIONS.index(fixed[i]) for i in idx])
    a=np.where(explore,rng.integers(0,3,n),f)
    prop=np.where(a==f,1-EPSILON+EPSILON/3,EPSILON/3)
    u=sample_usefulness(mu[idx,a],rng)
    return idx,a,prop,u


def fit(X,a,r):
    W=np.zeros((3,X.shape[1]))
    for k in range(3):
        m=a==k
        if m.any(): W[k]=np.linalg.solve(X[m].T@X[m]+RIDGE*np.eye(X.shape[1]),X[m].T@r[m])
    return W


def greedy(W,X): return np.argmax(X@W.T,axis=1)


def doubly_robust(W_model,X,a,prop,r,pi):
    """Per-sample DR value of a deterministic policy pi (array of actions) on logged data."""
    pred=X@W_model.T; n=np.arange(len(X))
    return pred[n,pi]+(a==pi)/prop*(r-pred[n,a])


def ips(X,a,prop,r,pi): return (a==pi)/prop*r


def gate(X,a,prop,r,fixed_a,rng,draws=1000):
    """Safe policy improvement: learn on half A, certify (learned - fixed) > 0 on half B with DR."""
    half=len(X)//2; A,B=slice(0,half),slice(half,None)
    W=fit(X[A],a[A],r[A]); pi=greedy(W,X[B])
    model=fit(X[A],a[A],r[A])
    diff=doubly_robust(model,X[B],a[B],prop[B],r[B],pi)-doubly_robust(model,X[B],a[B],prop[B],r[B],fixed_a[B])
    boots=[diff[rng.integers(0,len(diff),len(diff))].mean() for _ in range(draws)]
    low,high=np.percentile(boots,[2.5,97.5])
    return W,{'adopted':bool(low>0),'dr_delta':float(diff.mean()),'ci':[float(low),float(high)],'certified_on':int(len(diff)),'fitted_on':half}


# ------------------------------------------------------------------------------------ contexts

def contexts(res,quick=False):
    """Real decision contexts for deals where the learnable actions apply (no blocking issue, a score exists)."""
    from src.agent.thresholds import beliefs,decide
    rows=[]
    for pool,table in [('validation_replay',res.validation_replay),('test_replay',res.replay),('open',res.open)]:
        ids=sorted(table.index)[::(25 if quick else 1)]
        for oid in ids:
            st=beliefs(res,oid,pool); b=st['beliefs']
            if 'prediction' not in b or (b.get('data_quality') or {}).get('blocking'): continue
            fixed=decide(st,res.card,policy.DEFAULT_THRESHOLDS)
            if fixed not in ACTIONS: continue
            c=context_from(b,res.card); rows.append({'opportunity_id':oid,'pool':pool,'fixed':fixed,**c})
    frame=pd.DataFrame(rows)
    # Deterministic split by deal ID: 60% of deals can appear in feedback logs, 40% are held out for evaluation.
    frame['held_out']=frame.opportunity_id.map(lambda o:int(hashlib.sha256(o.encode()).hexdigest(),16)%10>=6)
    return frame


def evaluate(env,frame,quick=False):
    log=frame.loc[~frame.held_out]; held=frame.loc[frame.held_out]
    Xl=np.stack([phi(r) for r in log.to_dict('records')]); Xh=np.stack([phi(r) for r in held.to_dict('records')])
    fl=log.fixed.to_list(); fh=np.array([ACTIONS.index(f) for f in held.fixed])
    mu_l=environment(env,Xl,fl); mu_h=environment(env,Xh,held.fixed.to_list())
    true_h=expected_reward(mu_h); n=np.arange(len(Xh))
    v_fixed=true_h[n,fh].mean(); v_oracle=true_h.max(1).mean(); v_random=true_h.mean(1).mean()
    rows=[]
    for size in (LOG_SIZES[:3] if quick else LOG_SIZES):
        for seed in (SEEDS[:4] if quick else SEEDS):
            rng=np.random.default_rng(1000*seed+size)
            idx,a,prop,u=log_feedback(Xl,fl,mu_l,size,rng); r=np.array([reward(x) for x in u])
            X=Xl[idx]; fixed_a=np.array([ACTIONS.index(fl[i]) for i in idx])
            W_all=fit(X,a,r); learned=greedy(W_all,Xh)
            W_cert,cert=gate(X,a,prop,r,fixed_a,rng,200 if quick else 1000)
            gated=greedy(W_cert,Xh) if cert['adopted'] else fh
            pi_log=greedy(W_all,X)
            rows.append({'environment':env,'feedback_records':size,'seed':seed,'v_fixed':v_fixed,'v_learned':true_h[n,learned].mean(),
                         'v_gated':true_h[n,gated].mean(),'v_oracle':v_oracle,'v_uniform':v_random,'adopted':cert['adopted'],
                         'gate_dr_delta':cert['dr_delta'],'gate_ci_low':cert['ci'][0],
                         'ope_dr_learned':float(doubly_robust(W_all,X,a,prop,r,pi_log).mean()),'ope_ips_learned':float(ips(X,a,prop,r,pi_log).mean()),
                         'true_learned_on_log_contexts':float(expected_reward(mu_l[idx])[np.arange(len(idx)),pi_log].mean()),
                         'decisions_changed':float((gated!=fh).mean()),
                         'directional_without_evidence_rule':float(np.isin(gated,[0,2])[~np.isin(fh,[0,2])].mean()) if (~np.isin(fh,[0,2])).any() else 0.})
    return pd.DataFrame(rows)


def to_record(oid,c,action,default,prop,u):
    return FeedbackRecord(task_id=f'sim-{oid}',opportunity_id=oid,agent_decision=action,default_decision=default,propensity=float(prop),
                          context={k:float(v) for k,v in c.items()},reviewer_role='simulated reviewer',reviewer_action='followed',
                          usefulness=int(u),source='synthetic_simulator',notes='SYNTHETIC: drawn from src.agent.learning environment')


def demonstrate(frame,folder,env='E2_shifted_preferences',size=1000,seed=0):
    """The full path a pilot would use: feedback store -> verify chain -> learn -> certify -> saved policy file."""
    log=frame.loc[~frame.held_out].reset_index(drop=True)
    Xl=np.stack([phi(r) for r in log.to_dict('records')]); mu=environment(env,Xl,log.fixed.to_list())
    rng=np.random.default_rng(seed); idx,a,prop,u=log_feedback(Xl,log.fixed.to_list(),mu,size,rng)
    store_path=folder/f'synthetic_feedback_{env}.jsonl'
    if store_path.exists(): store_path.unlink()
    store=FeedbackStore(store_path)
    store.extend([to_record(log.opportunity_id[i],{k:log.loc[i,k] for k in ('distance_to_threshold','borderline','evidence_quality','conflict','disagreement_share','stale','out_of_range','blocking')},
                            ACTIONS[k],log.fixed[i],p,x) for i,k,p,x in zip(idx,a,prop,u)])
    chain=store.verify()
    policy_file=learn_from_store(store,'synthetic_simulator',folder/f'learned_policy_{env}.json',env,rng)
    return {'store':store_path.name,'chain':chain,'policy':policy_file}


def learn_from_store(store,source,path,label,rng=None):
    """Learn and certify from one source of a feedback store; write a JSON policy (no pickle) with its certificate."""
    entries=store.records(source)
    if not entries: return {'status':'no_feedback','source':source,'records':0,
                            'decision':'the agent keeps its fixed policy: there is no feedback to learn from'}
    rng=rng or np.random.default_rng(0)
    X=np.stack([phi({**{k:0. for k in FEATURES},**e['context']}) for e in entries])
    a=np.array([ACTIONS.index(e['agent_decision']) for e in entries]); prop=np.array([e['propensity'] for e in entries])
    r=np.array([reward(e['usefulness']) for e in entries])
    # The certificate compares against the policy that would have acted without exploration (default_decision).
    fixed_a=np.array([ACTIONS.index(e.get('default_decision') or e['agent_decision']) for e in entries])
    W,cert=gate(X,a,prop,r,fixed_a,rng)
    out={'status':'learned','source':source,'environment':label,'records':len(entries),'actions':ACTIONS,'features':FEATURES,
         'weights':{k:W[i].round(10).tolist() for i,k in enumerate(ACTIONS)},'certificate':cert,
         'policy_version':policy.POLICY_VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),
         'scope':'applies only to non-blocked, scored deals; blocked records and external actions keep the fixed safety rules',
         'warning':'SYNTHETIC feedback: demonstrates the learning mechanism, not real reviewer preferences or sales outcomes' if source=='synthetic_simulator' else ''}
    write_json(path,out)
    return {k:v for k,v in out.items() if k!='weights'}|{'file':Path(path).name}


def choose(learned,context):
    """Action of a saved, certified policy for one context; None when the certificate did not pass."""
    if not learned.get('certificate',{}).get('adopted'): return None
    x=phi({**{k:0. for k in FEATURES},**context}); scores={a:float(np.dot(learned['weights'][a],x)) for a in learned['actions']}
    return max(learned['actions'],key=lambda a:(scores[a],-learned['actions'].index(a)))


def figure(results,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluation.figures import save
    ink='#0E3B43'
    envs=list(ENVIRONMENTS); fig,axes=plt.subplots(1,len(envs),figsize=(15,3.9),sharey=False)
    for ax,env in zip(axes,envs):
        g=results.loc[results.environment.eq(env)].groupby('feedback_records')
        for col,label,color,ls in [('v_oracle','oracle (knows the simulator)','#9AA7B2',':'),('v_fixed','fixed agent policy','#5B7083','--'),
                                   ('v_learned','learned, no gate','#E39B2E','-'),('v_gated','learned + safety gate','#0F7A6C','-')]:
            m=g[col].mean(); lo=g[col].quantile(.025); hi=g[col].quantile(.975)
            ax.plot(m.index,m.values,ls,color=color,label=label,lw=2 if col=='v_gated' else 1.5,marker='o' if col in ('v_learned','v_gated') else None,ms=4)
            if col in ('v_learned','v_gated'): ax.fill_between(m.index,lo.values,hi.values,color=color,alpha=.12)
        ax.set_xscale('log'); ax.set_xlabel('synthetic feedback records'); ax.set_title(env.replace('_',' '),fontsize=10,color=ink,fontweight='bold')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('expected reviewer reward, held-out deals'); axes[-1].legend(fontsize=8,loc='lower right')
    fig.suptitle('SYNTHETIC environment: offline contextual-bandit learning from logged reviewer feedback (20 seeds, 2.5-97.5% band)',fontsize=11,color=ink)
    save(fig,out.figures/'feedback_learning.png')


def run(output_dir=DEFAULT_OUTPUT,quick=False):
    from src.agent.tools import Resources
    start=perf_counter(); out=Outputs(output_dir); folder=out.learning; folder.mkdir(parents=True,exist_ok=True)
    res=Resources(out.root)
    print('LEARNING: building real decision contexts (outcomes masked)...',flush=True)
    frame=contexts(res,quick); frame.to_csv(folder/'contexts.csv',index=False)
    results=pd.concat([evaluate(env,frame,quick) for env in ENVIRONMENTS],ignore_index=True)
    results.to_csv(folder/'learning_runs.csv',index=False)
    summary=results.groupby(['environment','feedback_records']).agg(
        v_fixed=('v_fixed','mean'),v_oracle=('v_oracle','mean'),v_learned=('v_learned','mean'),v_gated=('v_gated','mean'),
        gain_gated=('v_gated',lambda s:float((s-results.loc[s.index,'v_fixed']).mean())),
        gain_gated_low=('v_gated',lambda s:float(np.percentile(s-results.loc[s.index,'v_fixed'],2.5))),
        gain_gated_high=('v_gated',lambda s:float(np.percentile(s-results.loc[s.index,'v_fixed'],97.5))),
        regret_learned_runs=('v_learned',lambda s:float((s<results.loc[s.index,'v_fixed']-1e-12).mean())),
        regret_gated_runs=('v_gated',lambda s:float((s<results.loc[s.index,'v_fixed']-1e-12).mean())),
        adoption_rate=('adopted','mean'),decisions_changed=('decisions_changed','mean'),
        ope_dr_abs_error=('ope_dr_learned',lambda s:float((s-results.loc[s.index,'true_learned_on_log_contexts']).abs().mean())),
        ope_ips_abs_error=('ope_ips_learned',lambda s:float((s-results.loc[s.index,'true_learned_on_log_contexts']).abs().mean())),
        directional_without_evidence_rule=('directional_without_evidence_rule','mean'),seeds=('seed','count')).reset_index()
    summary.to_csv(folder/'learning_summary.csv',index=False)
    demo=demonstrate(frame,folder)
    real=learn_from_store(FeedbackStore(DEFAULT_STORE),'human_reviewer',folder/'learned_policy_human.json','real reviewer feedback')
    figure(results,out)
    result={'status':'complete','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full','elapsed_seconds':perf_counter()-start,
            'contexts':{'deals':len(frame),'held_out':int(frame.held_out.sum()),'by_pool':frame.pool.value_counts().to_dict(),
                        'fixed_policy_mix':frame.fixed.value_counts().to_dict()},
            'environments':ENVIRONMENTS,'epsilon':EPSILON,'sigma':SIGMA,'ridge':RIDGE,'actions':ACTIONS,'features':FEATURES,
            'demonstration':demo,'real_feedback':real,
            'claim_boundary':'Rewards are simulated. Results show that the learning, evaluation and gating code works when feedback exists; '
                             'they are not evidence about real reviewers, conversions or revenue.'}
    write_json(folder/'learning.json',result)
    print(f"LEARNING COMPLETE in {result['elapsed_seconds']:.0f}s: {len(frame)} contexts; real human feedback records: {real.get('records',0)}",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--quick',action='store_true',help='Subsampled contexts, 4 seeds; not reportable')
    args=parser.parse_args(); run(args.output,quick=args.quick)


if __name__=='__main__': main()
