"""Prospective pilot: how many deals and reviewers would be needed to measure business impact?

    python -m src.agent.pilot [--output RUN] [--quick]

Supports docs/PILOT_PROTOCOL.md. Nothing here measures impact: no pilot has run. Every assumption is an explicit
argument. The one empirical input is the between-agent variation of win rates in the TRAINING period (intra-class
correlation), used for the cluster-randomized design; base win rates come from development data.

Outputs (RUN/pilot/): power_analysis.csv (closed-form sample sizes), power_simulation.csv (Monte Carlo check with
agent-level random effects), pilot.json (assumptions and headline numbers).
"""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
import pandas as pd
from scipy.stats import norm
from src.data.crm import build
from src.data.split import asof_split
from src.outputs import DEFAULT_OUTPUT,Outputs,write_json

ALPHA=.05
POWER=.80
UPLIFTS=[.02,.03,.05,.08,.10]                 # absolute win-rate differences a pilot might target
SEED=7


def n_two_proportions(p1,p2,alpha=ALPHA,power=POWER):
    """Deals per arm for a two-sided two-proportion z-test (normal approximation)."""
    za,zb=norm.ppf(1-alpha/2),norm.ppf(power); pbar=(p1+p2)/2
    return int(np.ceil((za*np.sqrt(2*pbar*(1-pbar))+zb*np.sqrt(p1*(1-p1)+p2*(1-p2)))**2/(p2-p1)**2))


def icc(frame,group='sales_agent'):
    """One-way ANOVA estimate of the intra-class correlation of the binary outcome within agents."""
    g=frame.groupby(group).is_won; k=g.ngroups; n=len(frame); sizes=g.size()
    n0=(n-(sizes**2).sum()/n)/(k-1)
    ssb=(sizes*(g.mean()-frame.is_won.mean())**2).sum(); ssw=(sizes*g.mean()*(1-g.mean())).sum()
    msb=ssb/(k-1); msw=ssw/(n-k)
    return float(max(0.,(msb-msw)/(msb+(n0-1)*msw))),int(k),float(sizes.mean())


def simulate(p0,uplift,agents,deals_per_agent,rho,reps,rng):
    """Cluster-randomized by sales agent: half the agents get the agent-assisted workflow."""
    sd=np.sqrt(rho*p0*(1-p0)); hits=0
    for _ in range(reps):
        arm=rng.permutation(np.r_[np.zeros(agents//2),np.ones(agents-agents//2)])
        base=np.clip(p0+rng.normal(0,sd,agents),.01,.99)
        p=np.clip(base+uplift*arm,.01,.99)
        wins=rng.binomial(deals_per_agent,p)
        rate=wins/deals_per_agent
        # analysis at the cluster level (agent win rates), the unit of randomization: Welch t-test
        a,b=rate[arm==1],rate[arm==0]
        t=(a.mean()-b.mean())/np.sqrt(a.var(ddof=1)/len(a)+b.var(ddof=1)/len(b))
        hits+=abs(t)>norm.ppf(1-ALPHA/2)
    return hits/reps


def run(output_dir=DEFAULT_OUTPUT,quick=False):
    start=perf_counter(); out=Outputs(output_dir); folder=out.pilot; folder.mkdir(parents=True,exist_ok=True)
    d=build(); s=asof_split(d); train=d.frame.loc[s.train]
    dev=d.frame.loc[d.frame.close_date<s.test_start]
    p0=float(dev.is_won.mean()); rho,agents,mean_size=icc(train)
    rows=[]
    for u in UPLIFTS:
        n=n_two_proportions(p0,p0+u)
        for r in sorted({0.,.01,rho,.05}):
            for m in [20,40,80]:
                deff=1+(m-1)*r
                rows.append({'base_win_rate':p0,'uplift':u,'deals_per_arm_individual':n,'icc':r,'deals_per_agent':m,
                             'design_effect':deff,'deals_per_arm_cluster':int(np.ceil(n*deff)),
                             'agents_per_arm_needed':int(np.ceil(n*deff/m)),'agents_available':agents})
    table=pd.DataFrame(rows); table.to_csv(folder/'power_analysis.csv',index=False)
    rng=np.random.default_rng(SEED); sims=[]; reps=200 if quick else 2000
    for u in [.03,.05,.10]:
        for m in [40,80]:
            sims.append({'uplift':u,'agents':agents,'deals_per_agent':m,'icc':rho,'simulated_power':simulate(p0,u,agents,m,rho,reps,rng),
                         'replications':reps})
    sims=pd.DataFrame(sims); sims.to_csv(folder/'power_simulation.csv',index=False)
    headline=table.loc[(table.uplift.eq(.05))&(table.icc.eq(rho))&(table.deals_per_agent.eq(40))].iloc[0]
    result={'status':'complete','elapsed_seconds':perf_counter()-start,'alpha':ALPHA,'power':POWER,
            'base_win_rate_development':p0,'icc_sales_agent_training':rho,'sales_agents':agents,'mean_training_deals_per_agent':mean_size,
            'open_engaging_deals_in_snapshot':int(len(d.extra['scorable_open_deals'])),
            'headline':{'uplift':.05,'deals_per_arm_individual':int(headline.deals_per_arm_individual),
                        'deals_per_arm_cluster_40_per_agent':int(headline.deals_per_arm_cluster),
                        'agents_per_arm_needed':int(headline.agents_per_arm_needed)},
            'assumptions':['deal outcome observed within the follow-up window for every enrolled deal',
                           'uplift is a constant absolute difference in win rate',
                           'between-agent variation equals the training-period ICC',
                           'no contamination between arms when randomizing by sales agent'],
            'claim_boundary':'Planning numbers for a future study. No pilot has been run; no uplift is claimed.'}
    write_json(folder/'pilot.json',result)
    print(f"PILOT POWER COMPLETE: base win rate {p0:.3f}, ICC {rho:.4f}; 5-pt uplift needs {result['headline']['deals_per_arm_individual']} deals per arm",flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args(); run(args.output,quick=args.quick)


if __name__=='__main__': main()
