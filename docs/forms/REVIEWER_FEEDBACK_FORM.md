# Reviewer feedback form (one per agent review task)

Fill in one form for each review task the agent created (`review_tasks/<task_id>.json`). Save answers as one row of `reviewer_feedback_template.csv`, then import them:

    python -m src.agent.feedback import-csv my_feedback.csv

A single task can also be entered directly:

    python -m src.agent.feedback add --task review_tasks/<task_id>.json --usefulness 4 --reviewer-action followed --minutes 6 --agreed yes

Do not write customer names, e-mail addresses, phone numbers or your own name anywhere on this form; entries that look like contact details are rejected.

| Field | Question | Allowed answers |
|---|---|---|
| `task_id` | Task ID printed on the task | copied from the task |
| `opportunity_id` | Deal ID | copied from the task |
| `agent_decision` | Action the agent proposed | ESCALATE_AT_RISK_REVIEW, REVIEW_UNCERTAIN, MONITOR_NO_TASK, DATA_COMPLETION, ABSTAIN |
| `propensity` | Leave 1.0 unless the task says it was an exploration task (then copy its probability) | 0–1 |
| `reviewer_role` | Your role, not your name | e.g. account executive, sales manager |
| `reviewer_action` | What did you do? | followed, modified, rejected, data_fixed, no_action |
| `usefulness` | How useful was this task for deciding what to do? | 1 = waste of time, 2 = little use, 3 = neutral, 4 = useful, 5 = very useful |
| `agreed_with_agent` | Did you agree with the proposed action? | yes / no |
| `minutes_spent` | Minutes spent on this task | 0–600 |
| `outcome` | Leave empty. Filled later from the CRM, not by you | won, lost, still_open |
| `notes` | What evidence was missing or wrong? (max 500 characters) | free text, no personal data |

Usefulness becomes the learning reward `(usefulness − 3) / 2`, a value from −1 to +1. Deal outcomes are never used as a reward for the agent's action.
