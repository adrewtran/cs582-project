# Prospective human-review pilot: protocol for future use

This pilot has **not** been run. The project has no sales team and no live CRM. This protocol states what a small, safe pilot would need to measure before anyone could claim business value. Planning numbers come from `python -m src.agent.pilot` (`reports/crm/final/pilot/`).

## 1. Question

Does giving sales reviewers the agent's review tasks (evidence, data-quality flags, uncertainty, suggested human checks) improve how open deals are handled, compared with the usual process with the same model score? We ask about workflow outcomes first and sales outcomes second.

## 2. Design

| Item | Plan |
|---|---|
| Unit of randomization | Sales agent (cluster). Deal-level randomization would let one person see both arms. |
| Arms | **Control:** current process plus the calibrated score (system A). **Treatment:** the agent's review queue and tasks (system C). |
| Eligibility | Engaging deals with an engage date and an owner. Deals with a blocking data issue enter the data-completion queue in both arms. |
| Exploration for learning | In the treatment arm, 20% of eligible deals receive a review action drawn uniformly from the safe set {escalate, uncertain review, monitor}, with the probability logged. This enables later off-policy learning (`src/agent/learning.py`). It must be approved by the sales manager because some deals will get a less-preferred action. |
| Safety | No customer contact, CRM write or third-party contact is automated. Every external step is a human decision. Reviewers can override any task. |
| Duration | Enrolment until the sample size is reached, then 120 days of follow-up, the longest observed sales cycle being 138 days. |
| Blinding | Not possible for reviewers. Outcome data come from the CRM, not from reviewers. |

## 3. Outcomes

| Type | Measure | Source | When |
|---|---|---|---|
| Primary (workflow) | Share of missing-account records completed within 14 days | CRM audit log | 14 days |
| Primary (workflow) | Reviewer-rated usefulness of each task (1–5) | Feedback form | At review |
| Secondary (workflow) | Minutes per review; share of tasks the reviewer agreed with; overrides | Feedback form | At review |
| Secondary (workflow) | Time from engagement to first recorded follow-up | CRM activity log | 30 days |
| Business (confirmatory only if powered) | Win rate of enrolled deals | CRM deal stage | 120 days |
| Business (exploratory) | Close value of won deals; cycle length | CRM | 120 days |
| Safety | External actions executed without approval (must be 0); data-privacy incidents | Agent traces, audit | Continuous |

## 4. Sample size

From `reports/crm/final/pilot/pilot.json` and `power_analysis.csv`:

- Detecting a **5-point absolute increase** in win rate from the development base rate (about 0.63), at α = 0.05 and 80% power, needs **about 1,410 deals per arm** with deal-level analysis.
- The training-period intra-class correlation across sales agents is close to 0, so the cluster design adds little in this dataset. A real team may differ, so a design effect is still budgeted.
- The snapshot has 1,589 open Engaging deals in total. A pilot powered for win rate would need several months of new deals. A short pilot can only be powered for the workflow outcomes.

The pilot is therefore staged:

1. **Stage 1 (4–6 weeks):** workflow outcomes and feedback collection only, with no win-rate claim.
2. **Stage 2:** run only if Stage 1 shows usefulness and no safety issue. Continue enrolment until the business-outcome sample size is reached.

## 5. Analysis plan

- Pre-register outcomes and analysis before enrolment.
- Analyze at the cluster level, by agent-level means or a mixed model with an agent random effect.
- Report 95% CIs.
- Intention-to-treat: deals stay in their arm even when a reviewer ignores the task.
- Report all outcomes, including null and negative ones.
- No interim peeking at win rates; Stage 1 reviews workflow and safety only.

## 6. Feedback and learning

Each reviewed task produces one record in the local feedback store (`python -m src.agent.feedback add ...` or `import-csv` from the form). Records are validated and hash-chained, and they contain no reviewer names or customer contact details.

After Stage 1, `src.agent.learning.learn_from_store` fits a review policy on human feedback only. The agent may use it only if the safe-improvement certificate passes, i.e. the doubly-robust lower bound of the gain over the fixed policy is above 0. Simulated records are never mixed with human records.

## 7. What would and would not be claimed

| Evidence | Claim allowed |
|---|---|
| Stage 1 usefulness ratings and completion rates | Reviewers found tasks useful / data was fixed faster *in this pilot* |
| Stage 2 powered win-rate difference with CI excluding 0 | A causal effect of the workflow on win rate *in this team and period* |
| Anything from `src.agent.learning` (synthetic) | Only that the learning and gating code works as specified |
| Offline replay on historical deals | Association only; never uplift |

## 8. Ethics and data

- Data stay local; no customer is contacted by the system.
- Agent-level results are not used for individual performance evaluation.
- Reviewers can opt out.
- Records are kept only for the study period.
