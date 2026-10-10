"""Reviewer feedback: a validated, append-only, tamper-evident local store.

    python -m src.agent.feedback add --task TASK.json --usefulness 4 --reviewer-action followed [--store PATH]
    python -m src.agent.feedback import-csv FORM.csv [--store PATH]
    python -m src.agent.feedback verify [--store PATH]
    python -m src.agent.feedback summary [--store PATH]

The CRM dataset records no actions, no reviewer responses and no rewards, so the agent has nothing real to learn
from. This module is the collection mechanism a pilot would use (docs/PILOT_PROTOCOL.md,
docs/forms/REVIEWER_FEEDBACK_FORM.md). Each record links one agent decision (task id, decision, the probability
that the logging policy chose it, and the decision context) to what a human reviewer did and how useful the task
was. Records are hash-chained: editing or deleting a line breaks verification. Simulated records must say
source='synthetic_simulator' and are never mixed with human feedback when learning.

Privacy: no reviewer names (a role only), no customer contact details. Notes that look like an e-mail address or
a phone number are refused.
"""
import argparse
from dataclasses import asdict,dataclass,field
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
from src.agent import policy
from src.outputs import ROOT

DEFAULT_STORE=ROOT/'reports/crm/feedback/feedback.jsonl'      # git-ignored: real feedback stays local
SOURCES=('human_reviewer','synthetic_simulator')
REVIEWER_ACTIONS=('followed','modified','rejected','data_fixed','no_action')
DECISIONS=tuple(policy.DECISIONS)
OUTCOMES=(None,'won','lost','still_open')
CONTEXT_KEYS=('distance_to_threshold','borderline','evidence_quality','conflict','disagreement_share','stale','out_of_range','blocking')
PII=re.compile(r'[\w.+-]+@[\w-]+\.[\w.]+|\+?\d[\d\s().-]{7,}\d')


def reward(usefulness):
    """Reviewer usefulness 1..5 -> reward in [-1, 1]; the only reward the learner may use."""
    return (int(usefulness)-3)/2


@dataclass
class FeedbackRecord:
    task_id: str
    opportunity_id: str
    agent_decision: str
    propensity: float                 # probability that the logging policy chose agent_decision (1.0 = deterministic)
    context: dict                     # CONTEXT_KEYS, from the agent's own evidence (no outcome)
    reviewer_role: str                # e.g. 'account executive', never a name
    reviewer_action: str
    usefulness: int                   # 1 (waste of time) .. 5 (very useful)
    source: str='human_reviewer'
    default_decision: str|None=None   # what the fixed policy would have done (differs only when the logger explored)
    policy_version: str=policy.POLICY_VERSION
    agreed_with_agent: bool|None=None
    minutes_spent: float|None=None
    outcome: str|None=None            # filled later, only from the CRM, never by the agent
    notes: str=''
    created_utc: str=field(default_factory=lambda:datetime.now(timezone.utc).isoformat())

    def validate(self):
        errors=[]
        if not self.task_id or not self.opportunity_id: errors.append('task_id and opportunity_id are required')
        if self.agent_decision not in DECISIONS: errors.append(f'agent_decision must be one of {DECISIONS}')
        if self.default_decision not in (None,)+DECISIONS: errors.append(f'default_decision must be one of {DECISIONS}')
        if not 0<self.propensity<=1: errors.append('propensity must be in (0, 1]')
        if set(self.context)-set(CONTEXT_KEYS): errors.append(f'unknown context keys {sorted(set(self.context)-set(CONTEXT_KEYS))}')
        if self.reviewer_action not in REVIEWER_ACTIONS: errors.append(f'reviewer_action must be one of {REVIEWER_ACTIONS}')
        if not isinstance(self.usefulness,int) or not 1<=self.usefulness<=5: errors.append('usefulness must be an integer 1..5')
        if self.source not in SOURCES: errors.append(f'source must be one of {SOURCES}')
        if self.outcome not in OUTCOMES: errors.append(f'outcome must be one of {OUTCOMES}')
        if self.minutes_spent is not None and not 0<=float(self.minutes_spent)<=600: errors.append('minutes_spent must be 0..600')
        if len(self.notes)>500: errors.append('notes are limited to 500 characters')
        if PII.search(self.notes) or PII.search(self.reviewer_role): errors.append('notes/role look like contact details; remove them')
        if errors: raise ValueError('; '.join(errors))
        return self


def context_from(beliefs,card=None,th=policy.DEFAULT_THRESHOLDS):
    """Decision context from the agent's evidence. Uses no outcome field."""
    from src.agent.planner import assess
    p=beliefs.get('prediction') or {}; q=beliefs.get('data_quality') or {}; g=beliefs.get('agreement') or {}
    a=beliefs.get('assessment') or (assess(beliefs,card,th) if card and p else None)
    d=float(p.get('distance_to_threshold',0.))
    warnings=set(q.get('warnings',[]))
    return {'distance_to_threshold':round(d,10),'borderline':float(abs(d)<th.borderline_margin),
            'evidence_quality':float((a or {}).get('evidence_quality',0.)),'conflict':float(bool((a or {}).get('conflicts'))),
            'disagreement_share':float(g.get('disagreement_share',0.)),'stale':float('stale_open_record' in warnings),
            'out_of_range':float('outside_training_range' in warnings),'blocking':float(bool(q.get('blocking')))}


class FeedbackStore:
    def __init__(self,path=DEFAULT_STORE,guard=None):
        self.path=Path(path)
        if guard is not None: self.path=guard.local_path(self.path)

    @staticmethod
    def digest(entry):
        body={k:v for k,v in entry.items() if k!='record_hash'}
        return hashlib.sha256(json.dumps(body,sort_keys=True,ensure_ascii=False).encode()).hexdigest()

    def entries(self):
        if not self.path.exists(): return []
        return [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]

    def append(self,record):
        record.validate(); entries=self.entries()
        entry={**asdict(record),'previous_hash':entries[-1]['record_hash'] if entries else 'GENESIS'}
        entry['record_hash']=self.digest(entry)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f: f.write(json.dumps(entry,ensure_ascii=False)+'\n')
        return entry

    def extend(self,records):
        """Append many records in one write (same chain rule as append)."""
        entries=self.entries(); previous=entries[-1]['record_hash'] if entries else 'GENESIS'; lines=[]
        for record in records:
            record.validate(); entry={**asdict(record),'previous_hash':previous}
            entry['record_hash']=previous=self.digest(entry); lines.append(json.dumps(entry,ensure_ascii=False))
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with self.path.open('a',encoding='utf-8') as f: f.write('\n'.join(lines)+('\n' if lines else ''))
        return len(lines)

    def verify(self):
        previous='GENESIS'
        for i,entry in enumerate(self.entries(),1):
            if entry.get('previous_hash')!=previous or self.digest(entry)!=entry.get('record_hash'):
                return {'ok':False,'records':i-1,'broken_at':i}
            previous=entry['record_hash']
        return {'ok':True,'records':len(self.entries())}

    def records(self,source):
        if source not in SOURCES: raise ValueError(source)
        return [e for e in self.entries() if e['source']==source]

    def summary(self):
        entries=self.entries(); out={'records':len(entries),'chain':self.verify()}
        for source in SOURCES:
            rows=[e for e in entries if e['source']==source]
            out[source]={'records':len(rows),'mean_usefulness':sum(e['usefulness'] for e in rows)/len(rows) if rows else None,
                         'by_decision':{d:sum(e['agent_decision']==d for e in rows) for d in DECISIONS if any(e['agent_decision']==d for e in rows)}}
        return out


def from_task(task,usefulness,reviewer_action,reviewer_role='sales reviewer',**extra):
    """Build a human feedback record from a review task written by the agent (review_tasks/*.json)."""
    ev=task['evidence']; p=ev.get('win_probability'); t=ev.get('decision_threshold')
    context={'distance_to_threshold':round(p-t,10) if p is not None and t is not None else 0.,
             'evidence_quality':float(ev.get('evidence_quality') or 0.),'conflict':float(bool(ev.get('conflicts'))),
             'blocking':float(task['decision']=='DATA_COMPLETION')}
    return FeedbackRecord(task_id=task['task_id'],opportunity_id=task['opportunity_id'],agent_decision=task['decision'],
                          propensity=1.0,context=context,reviewer_role=reviewer_role,reviewer_action=reviewer_action,
                          usefulness=int(usefulness),**extra)


def import_csv(path,store):
    """The paper/online form (docs/forms/reviewer_feedback_template.csv) -> validated records."""
    import csv
    records=[]
    with open(path,newline='',encoding='utf-8') as f:
        for i,row in enumerate(csv.DictReader(f),2):
            try:
                records.append(FeedbackRecord(task_id=row['task_id'],opportunity_id=row['opportunity_id'],agent_decision=row['agent_decision'],
                    propensity=float(row.get('propensity') or 1),context={},reviewer_role=row['reviewer_role'],
                    reviewer_action=row['reviewer_action'],usefulness=int(row['usefulness']),
                    agreed_with_agent={'yes':True,'no':False}.get(row.get('agreed_with_agent','').strip().lower()),
                    minutes_spent=float(row['minutes_spent']) if row.get('minutes_spent') else None,
                    outcome=row.get('outcome') or None,notes=row.get('notes','')).validate())
            except (KeyError,ValueError) as exc: raise ValueError(f'{path} line {i}: {exc}') from exc
    return store.extend(records)


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub=parser.add_subparsers(dest='command',required=True)
    add=sub.add_parser('add'); add.add_argument('--task',type=Path,required=True)
    add.add_argument('--usefulness',type=int,required=True); add.add_argument('--reviewer-action',required=True,choices=REVIEWER_ACTIONS)
    add.add_argument('--reviewer-role',default='sales reviewer'); add.add_argument('--minutes',type=float)
    add.add_argument('--agreed',choices=['yes','no']); add.add_argument('--notes',default='')
    imp=sub.add_parser('import-csv'); imp.add_argument('csv',type=Path)
    sub.add_parser('verify'); sub.add_parser('summary')
    for p in [add,imp]+[sub.choices['verify'],sub.choices['summary']]: p.add_argument('--store',type=Path,default=DEFAULT_STORE)
    args=parser.parse_args(); store=FeedbackStore(args.store)
    if args.command=='add':
        task=json.loads(args.task.read_text(encoding='utf-8'))
        entry=store.append(from_task(task,args.usefulness,args.reviewer_action,args.reviewer_role,minutes_spent=args.minutes,
                                     agreed_with_agent={'yes':True,'no':False}.get(args.agreed),notes=args.notes))
        print(f"recorded {entry['task_id']} (usefulness {entry['usefulness']}, reward {reward(entry['usefulness']):+.1f}); chain {store.verify()}")
    elif args.command=='import-csv': print(f'imported {import_csv(args.csv,store)} records; chain {store.verify()}')
    elif args.command=='verify': print(json.dumps(store.verify()))
    else: print(json.dumps(store.summary(),indent=2))


if __name__=='__main__': main()
