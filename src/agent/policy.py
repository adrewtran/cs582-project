"""The agent's explicit, versioned policy: tool costs, information needs, evidence weights and permissions.

Every number the agent uses to choose a step or a final action is defined here, so the decision rule can be
read, cited and ablated. None of these constants was tuned on test outcomes; they are fixed design choices.
"""
from dataclasses import dataclass
from pathlib import Path

POLICY_VERSION='evidence-agent-1.0'
MAX_STEPS=14                 # hard step budget per opportunity (observe + act + report)
MIN_GAIN=0.10                # a tool runs only when need - cost >= MIN_GAIN
MAX_RETRIES=1                # a failed tool is retried once, then the agent degrades gracefully
BORDERLINE_MARGIN=0.05       # |calibrated P(Won) - decision threshold| below this is "borderline"
SUPPORT_MIN=5                # prior closed deals needed before an entity's history counts as evidence
CONFLICT_GAP=0.10            # prior account win rate this far from the archive rate conflicts with the model
COMMITTEE_SPLIT=0.40         # share of committee models on the other side of their validation median
SUFFICIENT_EVIDENCE=0.70     # evidence-quality score needed for an escalation or a no-task decision
STALE_QUANTILE=0.95          # open longer than this quantile of training sales cycles -> stale record



@dataclass(frozen=True)
class Thresholds:
    """Decision-policy thresholds of the agent, kept apart from the model's classification threshold.

    The model's threshold (models/model_card.json: decision_threshold) is selected on validation by macro-F1
    in src.train. These six values are policy design choices; src.agent.thresholds tests whether any of them
    can be chosen by an objective on validation data. Safety rules (blocking data issues never get a
    directional action; external tools never run) are not thresholds and cannot be changed here.
    """
    min_gain: float=MIN_GAIN
    borderline_margin: float=BORDERLINE_MARGIN
    support_min: int=SUPPORT_MIN
    conflict_gap: float=CONFLICT_GAP
    committee_split: float=COMMITTEE_SPLIT
    sufficient_evidence: float=SUFFICIENT_EVIDENCE


DEFAULT_THRESHOLDS=Thresholds()

# Relative compute cost of each information tool (model calls dominate).
TOOL_COST={'get_opportunity':.02,'get_account_information':.02,'check_data_quality':.05,
           'get_prior_sales_history':.08,'predict_win_probability':.05,'check_model_agreement':.15,
           'explain_prediction':.15,'evaluate_evidence':.02}

# Evidence-quality checklist: weights sum to 1. Each item is verifiable from data or saved artifacts.
EVIDENCE_WEIGHTS={'account_present':.25,'account_history_supported':.15,'product_history_supported':.10,
                  'agent_history_supported':.10,'no_imputed_inputs':.15,'inputs_in_training_support':.15,
                  'committee_agrees':.10}

# Final actions, most specific first. The agent picks the first one whose requirements all hold
# ("the most informative action that the evidence can support").
DECISIONS=['ESCALATE_AT_RISK_REVIEW','MONITOR_NO_TASK','REVIEW_UNCERTAIN','DATA_COMPLETION','ABSTAIN']

# Permission classes. Only the first three may run autonomously.
AUTONOMOUS={'read','compute','local_write'}
REQUIRES_HUMAN={'external_write','communication'}


class PolicyViolation(PermissionError):
    """Raised when a tool call would leave the agent's permitted, local scope."""


class SafetyGuard:
    """Checks every tool call before execution. External effects are never executed, only escalated."""

    def __init__(self,output_root):
        self.output_root=Path(output_root).resolve()

    def check(self,tool,arguments):
        if tool.permission in REQUIRES_HUMAN:
            raise PolicyViolation(f'{tool.name} ({tool.permission}) needs human approval; the agent never executes it')
        if tool.permission not in AUTONOMOUS:
            raise PolicyViolation(f'{tool.name}: unknown permission {tool.permission}')

    def local_path(self,path):
        path=Path(path).resolve()
        if not path.is_relative_to(self.output_root):
            raise PolicyViolation(f'local writes are confined to {self.output_root}; refused {path}')
        return path
