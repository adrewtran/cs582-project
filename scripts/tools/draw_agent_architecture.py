"""Draw the agent architecture diagram used by the paper and slides -> docs/figures/agent_architecture.png."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT=Path(__file__).resolve().parents[2]
TEAL,ORANGE,GREY,DARK,LIGHT,RED='#0F7A6C','#E39B2E','#5B7083','#0E3B43','#EEF4F3','#D2554E'   # deck theme colours


def box(ax,x,y,w,h,title,body,color,fill='white',fs=8.6):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.018',lw=1.6,ec=color,fc=fill))
    ax.text(x+w/2,y+h-0.032,title,ha='center',va='top',fontsize=fs+1.4,fontweight='bold',color=DARK)
    ax.text(x+w/2,y+h-0.085,body,ha='center',va='top',fontsize=fs,color=DARK,linespacing=1.35)


def arrow(ax,a,b,color=DARK,text='',style='-|>',ls='-'):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle=style,color=color,lw=1.5,ls=ls,shrinkA=2,shrinkB=2))
    if text: ax.text((a[0]+b[0])/2,(a[1]+b[1])/2+0.012,text,ha='center',va='bottom',fontsize=7.6,color=color)


def main():
    fig,ax=plt.subplots(figsize=(13,6.6)); ax.set(xlim=(0,1),ylim=(0,1)); ax.axis('off')
    ax.text(0.5,0.985,'Evidence-grounded CRM decision agent: bounded, deterministic, tool-using',ha='center',va='top',fontsize=14,fontweight='bold',color=DARK)
    # Left: frozen ML pipeline (unchanged methodology)
    box(ax,0.015,0.50,0.2,0.38,'Frozen ML pipeline','src.train / evaluate / predict\n6 models x {raw, history}\nvalidation-only selection lock\ncalibrated P(Won) + threshold\nmodel card (validation only)',GREY,LIGHT)
    box(ax,0.015,0.10,0.2,0.32,'Real CRM data (read-only)','sales_pipeline, accounts,\nproducts, sales_teams\nfrozen training archive\n(outcomes masked for tools)',GREY,LIGHT)
    # Middle: the agent loop
    box(ax,0.27,0.13,0.42,0.75,'','',TEAL,'#fbfefe')
    ax.text(0.48,0.87,'Agent controller (max 14 steps)',ha='center',va='top',fontsize=11.5,fontweight='bold',color=TEAL)
    steps=[('1 Goal','triage one deal, or build a review\nqueue within a budget'),
           ('2 Observe','record, account, data quality'),
           ('3 Plan','utility = need - cost; evidence-tool need\n= value of information (lookahead)'),
           ('4 Act','call the chosen tool via the guard;\nretry once, never invent a result'),
           ('5 Evaluate','evidence quality, conflicts, committee,\nmodel reliability (validation CI)'),
           ('6 Decide','most specific action the evidence\nsupports (requirements listed)')]
    for i,(t,b) in enumerate(steps):
        y=0.735-i*0.1
        ax.add_patch(FancyBboxPatch((0.29,y),0.36,0.075,boxstyle='round,pad=0.006,rounding_size=0.012',lw=1,ec=TEAL,fc='white'))
        ax.text(0.30,y+0.0375,t,va='center',fontsize=9.2,fontweight='bold',color=TEAL)
        ax.text(0.395,y+0.0375,b,va='center',fontsize=7.9,color=DARK)
        if i<5: arrow(ax,(0.47,y),(0.47,y-0.019),TEAL)
    ax.annotate('',xy=(0.655,0.57),xytext=(0.655,0.37),arrowprops=dict(arrowstyle='-|>',color=TEAL,lw=1.4,ls='--',connectionstyle='arc3,rad=-0.6'))
    ax.text(0.47,0.165,'Steps 3-5 repeat until no tool clears the minimum gain (0.10), then 6 runs once.',ha='center',fontsize=8,color=TEAL,style='italic')
    box(ax,0.73,0.50,0.255,0.38,'Local tools (read / compute)','get_opportunity\nget_account_information\nget_prior_sales_history\ncheck_data_quality\npredict_win_probability\ncheck_model_agreement\nexplain_prediction, evaluate_evidence',TEAL,LIGHT,fs=8.2)
    box(ax,0.73,0.25,0.255,0.2,'Local writes (autonomous)','create_review_task -> outbox/*.json\nexport_agent_report -> trace + report',TEAL,'white',fs=8.2)
    box(ax,0.73,0.03,0.255,0.17,'External actions (never executed)','send_customer_email, update_crm_record,\ncontact_third_party -> blocked by guard,\nqueued for human approval',RED,'#fff5f2',fs=8.0)
    arrow(ax,(0.217,0.69),(0.268,0.69),DARK); ax.text(0.242,0.70,'frozen\nmodel',ha='center',va='bottom',fontsize=7.4)
    arrow(ax,(0.217,0.26),(0.268,0.26),DARK); ax.text(0.242,0.27,'records',ha='center',va='bottom',fontsize=7.4)
    arrow(ax,(0.692,0.69),(0.728,0.69),TEAL); arrow(ax,(0.692,0.35),(0.728,0.35),TEAL); arrow(ax,(0.692,0.115),(0.728,0.115),RED)
    ax.text(0.48,0.105,'Every step -> machine-readable trace: goal, candidates with utilities, chosen tool,\narguments, sources, result, evaluation, decision requirements,\nstopping reason, model hash and content hash.',ha='center',va='top',fontsize=8,color=DARK)
    out=ROOT/'docs/figures/agent_architecture.png'; out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(out,dpi=170,bbox_inches='tight'); plt.close(fig); print(out)


if __name__=='__main__': main()
