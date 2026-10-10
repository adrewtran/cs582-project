# Novelty and observed results

All numbers come from the verified run in `reports/crm/final/` (Linux, CPU, 2026-10-10, after merging main into PR #4). The agent design is specified in `docs/AGENT_ARCHITECTURE.md`. We keep three claims separate:

1. **Prediction quality:** how well the models rank won and lost deals. *Weak, near chance.*
2. **Agent workflow quality:** how reliably the system gathers evidence, avoids unsupported recommendations, acts safely and records its decisions. *Measured improvement over two baselines.*
3. **Business value:** whether sales outcomes improve. *Not measured. Not claimed.*

## 1. Prediction quality (real CRM data, exploratory test of 1,361 deals)

Six models (Dummy, LR, RF, MLP, TabNet, CatBoost) were each trained on two feature sets: **raw** (18 inputs) and **history** (raw plus strictly prior agent/account/product outcomes from the frozen training archive). All 12 candidates used one time-based split. The pair was locked on validation **before** any test metric existed (`models/selection_lock.json`).

| | Validation ROC-AUC | Test ROC-AUC |
|---|---|---|
| Selected: history / Logistic Regression | 0.5692 (95% CI 0.523–0.616) | **0.5168** |
| Raw / Logistic Regression | 0.5630 | 0.5238 |
| All 12 learned candidates | 0.496–0.569 | 0.480–0.542 |
| Raw / CatBoost (highest on test, **not** selectable after the fact) | 0.5446 | 0.5424 |

- History features did not improve discrimination. The test ROC-AUC change (history − raw) is −0.007 for LR, −0.039 for RF, +0.004 for MLP, +0.013 for TabNet and −0.044 for CatBoost (`metrics/history_auc_deltas.csv`).
- The Dummy gets 60.0% accuracy by always predicting Won. Sigmoid calibration lowers the selected model's test Brier from 0.3180 to 0.2489, still worse than the Dummy's 0.2421.
- The leakage control (a depth-one tree on `close_value`) reaches test ROC-AUC 1.0. It is excluded and shows why outcome fields must never be features.
- Split check: a random split raises ROC-AUC by at most 0.036 (`checks/split_protocol_summary.csv`). All protocols stay near chance.
- Explanation check on the raw Random Forest: reference sensitivity agrees with TreeSHAP (median Spearman 0.54; top-3 overlap 73% vs 8% by chance). Resetting the selected model's top-3 inputs moves P(Won) 6.3× more than 3 random inputs.
- **Known-truth simulation** (`data/crm_simulated/`, `SIMULATED_DATA.md`): the same pipeline reaches TabNet 0.869 vs an oracle 0.878. The planted formula is invented, so a high simulated AUC says the pipeline *can* learn when signal exists. It says nothing about real sales, and it never enters the real-data numbers.

**Why the real data is hard.** Win rates barely vary across products, sectors and agents. There is no deal-level engagement information (calls, budget, competitor). 1,088 of the 1,589 open deals (68%) lack an account, while no closed deal does. And the selected model's open-deal scores lean on `global_closed_count`, an input that grows with the calendar. For 827 open deals (52%), at least one input is outside the training range, mostly that count (2,975 vs a training maximum of 2,938).

**Differences from earlier runs.** The merged code reproduces PR #4's LR, RF, CatBoost and raw MLP numbers exactly, and the same selection (history/LR, test 0.5168). TabNet and the history MLP differ from PR #4's earlier run, which was made in a different environment: raw TabNet 0.4924 → 0.4893, history TabNet 0.4770 → 0.5024, history MLP 0.4987 → 0.4991. Main's earlier macOS run had RF 0.518 and TabNet 0.507 on raw features. Two consecutive full runs on the verification machine gave identical metrics for all 12 candidates and bit-identical open-deal scores. Only the multi-threaded Random Forest diagnostics varied, in the third decimal (TreeSHAP agreement median 0.545 vs 0.544). No conclusion changes.

## 2. Agent workflow quality (identical inputs, same frozen model)

**Systems:** **A** = the traditional prediction pipeline (`src.predict`: P(Won), label, priority band, factors, input warning). **B** = the PR #4 rule-based assistant (fixed IF/THEN rules). **C** = the evidence-grounded agent.

### All 1,589 Engaging open deals (`agent/system_comparison.csv`)

| Measure | A | B | C (agent) |
|---|---|---|---|
| Unsupported directional labels (out-of-support or borderline deals) | 1,242 (100%) | 487 (39.2%) | **0 (0%)** |
| Known data-issue recall (missing account, retrospective, stale) | 0.28 | 0.64 | **1.00** |
| Evidence types cited, of 7 (all deals / complete-account deals) | 4.0 / 4.0 | 5.0 / 5.0 | 4.86 / **6.73** |
| Cited claims re-verified from raw CSVs / bundle | 100% | 100% | 100% (17,538 claims) |
| Manual steps left, of 9 | 5.0 | 3.0 | **2.22** |
| Local review tasks created | 0 | 0 | 1,455 |
| External actions executed | 0 | 0 | 0 |
| Complete, replayable trace | no | no | **100%** |
| Mean runtime per deal | 551 ms* | 52 ms | 79 ms |

\*A uses the project's unbatched 19-call explanation. B and C share the batched implementation, so **B → C (+27 ms)** is the fair overhead.

The agent cites fewer evidence types on average than B because it deliberately does not explain or cross-check the 1,088 out-of-support scores. It routes them to data completion instead. Agent decisions: 1,088 `DATA_COMPLETION`, 338 `REVIEW_UNCERTAIN`, 134 `MONITOR_NO_TASK`, 29 `ESCALATE_AT_RISK_REVIEW`.

### Scenarios (`agent/scenario_results.csv`; expectations fixed in `src/agent/scenarios.py` before running)

| | S1 complete | S2 missing account | S3 borderline | S4 conflict | S5 batch | S6a–c invalid IDs | S6d tool failure | S7 prohibited request | Passed |
|---|---|---|---|---|---|---|---|---|---|
| A | ✓ | ✗ | ✗ | ✗ | ✗ | ✓✓✓ | ✗ | ✗ | 4/10 |
| B | ✓ | ✓ | ✓ | ✗ | ✗ | ✓✓✓ | ✗ | ✗ | 6/10 |
| C | ✓ | ✓ | ✓ | ✓ | ✓ | ✓✓✓ | ✓ | ✓ | 10/10 |

S6d (injected model failure) and S7 (a request to email the customer and update the CRM) are constructed on purpose. The other cases are real records chosen by documented rules. Several checks test capabilities the agent was built to add, so baselines fail them by construction. They show what the agent adds. They are not a contest it was tuned to win.

### Ablations on the same 1,589 deals (`agent/ablation_summary.csv`)

| Agent without… | Decisions changed | Unsupported label rate | Data-issue recall | Tool calls / deal | Runtime / deal |
|---|---|---|---|---|---|
| (full agent) | — | 0% | 1.00 | 6.09 | 79 ms |
| evidence retrieval | 10.3% | 0% | 1.00 | 5.32 | 70 ms |
| reliability checks | 80.1% | 53.1% | 0.00 | 4.80 | 37 ms |
| planning (fixed full sequence) | 0% | 0% | 1.00 | 8.23 | 229 ms |
| tool execution (dry run) | 0% | 0% | 1.00 | 5.18 | 76 ms |

Reading: the **reliability checks** produce the safety benefit. **Evidence retrieval** changes 10% of decisions; without it, 134 more deals go to uncertain review because support cannot be shown. **Planning** does not change decisions. It saves 2.1 tool calls and 65% of the runtime per deal. **Tool execution** turns decisions into 1,455 local tasks and removes 0.92 manual steps per deal.

**Reproducibility:** 200/200 re-runs on fresh resources gave identical trace hashes. 880/880 recorded tool calls replayed to identical results (`agent/experiment.json`).

## 3. Business value: offline outcome-masked replay (not uplift)

The agent was replayed on the 1,361 labelled test-period deals, with outcome columns removed before any tool could read them. Labels were joined only afterwards (`agent/replay_outcomes.csv`). Base win rate: 60.0%.

| Group | Deals | Observed win rate |
|---|---|---|
| Agent `ESCALATE_AT_RISK_REVIEW` | 289 | 62.3% |
| Agent `REVIEW_UNCERTAIN` | 1,049 | 59.0% |
| Agent `MONITOR_NO_TASK` | 23 | 78.3% |
| A/B lowest band (P(Won) < 0.40) | 38 | 39.5% |

The deals the agent escalated as at risk were **not** lost more often than average. The agent cannot rank deals better than the near-chance model it uses. This replay is an evaluation, not a learning loop. No sales uplift, causal effect or revenue gain is claimed.

## 4. Novelty contributions and their evidence

| Contribution | Implementation | Experimental evidence |
|---|---|---|
| Value-of-information tool planning (evidence gathered only if it can change the action or is needed by the reviewer) | `src/agent/planner.py` | No-planning ablation: same decisions, +2.1 tool calls, 2.9× runtime |
| Reliability-gated decisions (no confident label on out-of-support, borderline, conflicting or chance-level evidence) | `planner.decision_candidates`, `planner.assess` | Unsupported labels 0 vs 1,242 (A) and 487 (B); without the gate, 53% return |
| Strictly prior, source-cited evidence retrieval with independent re-verification | `src/agent/tools.py`, `src/data/history.py`, `experiment.Verifier` | 100% of 17,538 claims verified; 6.73 evidence types on complete records |
| Bounded autonomy with a permission guard and approval queue | `src/agent/policy.py`, `controller.act` | S7 passed; network-blocked test; 0 external actions |
| Replayable decision traces | `src/agent/controller.py` | 200/200 identical hashes; 880/880 tool replays |
| Budgeted multi-deal review planning | `controller.run_batch` | S5 passed: queue within budget; missing accounts kept out of the sales queue |
| Separation of prediction, workflow and business claims | this file; outcome-masked replay | Replay shows no outcome targeting gain; reported, not hidden |

**Not claimed:** a new learning algorithm, better ROC-AUC, online learning or reinforcement learning, an LLM, causal effects, validated priority bands, or sales or revenue uplift.

## Natural ESL answer to the professor

"We don't claim a new learning algorithm, and our prediction is weak. That is the honest result on this data. Our novelty is the agent. It decides which evidence to collect by asking if it could change the decision. It refuses to give a confident label when the evidence is weak. It acts on its own only inside safe limits, and it keeps a replayable trace. On the same 1,589 real deals, it gave zero unsupported labels, while the traditional pipeline gave 1,242. It does not make the model more accurate, and we did not measure more sales."
