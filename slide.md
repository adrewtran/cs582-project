# CRM_Final — slide content for review

Cách review: mỗi slide có **Layout** (bố cục / hình), **On slide** (chữ hiện trên slide), **Notes** (speaker notes) và **Source** (file chứa con số). Sửa trực tiếp file này, rồi cập nhật `CRM_Final.pptx` bằng tay cho khớp.

Phiên bản 2026-10-10 (PR #4 sau khi merge `main`): 21 slide chính + 3 slide dự phòng (backup). Mọi con số đã được đối chiếu với `reports/crm/final/` (lần chạy Linux ngày 2026-10-10) và `data/crm_simulated/`; `tests/test_documents.py` kiểm tra tự động các con số chính. Thay đổi so với bản 18 slide trước:
- Sáu model (thêm CatBoost) × hai bộ đặc trưng (raw, history), chọn trên validation và khóa trước khi xem test; model được chọn: history/Logistic Regression (test ROC-AUC 0.5168).
- Thêm phần agent (slide 12–19): giới hạn của ML truyền thống, kiến trúc agent, so sánh A/B/agent, quy trình trên một deal thật, demo trực tiếp (`python -m src.agent_demo`), thí nghiệm baseline và ablation, đóng góp mới.
- Slide "How MLP and TabNet work", "TabNet vs MLP" và section "Scenario B" chuyển xuống phần backup để giữ thời lượng; số liệu đã cập nhật.
- Phân vai: Speaker 1 slide 1–7, Speaker 2 slide 8–14, Speaker 3 (điều khiển demo) slide 15–21; lời thoại đầy đủ ở `docs/ESL_PRESENTATION_TRANSCRIPT.md`.

---

## Slide 1 — Title

**Layout:** Title slide.

**On slide:**
- Title: Predicting CRM Sales Opportunities Using Machine Learning
- Subtitle: CS 582 Machine Learning · Group 2 · Final presentation
- Hong Thai Phan · Nguyen Khanh An Tran · Hoang Thien Bao Bui

**Notes:** Hello everyone. We are Group 2. Our project is "Predicting CRM Sales Opportunities Using Machine Learning". We predict whether a sales deal will be won or lost. Then we built an agent that uses this prediction carefully and safely.

---

## Slide 2 — Goal: a win probability and its reasons

**Layout:** Three cards in a row (Input → Models → Output), then a two-line box for the two scenarios.

**On slide:**
- **Input:** Four CRM tables: sales pipeline, accounts, products, sales teams
- **Models:** Dummy control, Logistic Regression, Random Forest, MLP, TabNet, CatBoost; raw and history features
- **Output:** P(Won) for each open deal, the factors that move its score, and an agent that checks the evidence and decides the next safe step
- **Two scenarios, same pipeline**
  - A · Original Maven CRM data: the project's reported results.
  - B · Simulated data with a known formula: a check that the pipeline finds signal when signal exists.

**Notes:** Our input is four CRM tables: the sales pipeline, accounts, products and sales teams. We train six models. The output is a win probability for each open deal and the factors behind it. On top of that, an agent collects evidence, checks it, and decides the next safe step. We use two data scenarios. Scenario A is the real CRM data, and these are our reported results. Scenario B is simulated data with a known formula. We only use it to check that our pipeline can learn when there is a signal.

---

## Slide 3 — Data: a fictitious B2B hardware seller

**Layout:** Bar chart on the left (deal stage of all 8,800 opportunities); four stat callouts on the right.

**Chart — Deal stage of all 8,800 opportunities:** Won 4,238 · Lost 2,473 · Engaging (open) 1,589 · Prospecting (open) 500

**On slide:**
- 8,800 sales opportunities · 85 customer accounts · 7 products · 35 sales agents
- Train only on 6,711 closed deals (63% Won). Open deals are scored, never labelled Lost.
- Source: Maven Analytics CRM Sales Opportunities (public practice dataset)

**Notes:** The data is the Maven Analytics CRM dataset. It is a fictitious company that sells computer hardware to other businesses. It has 8,800 deals, 85 accounts, 7 products and 35 sales agents. We train only on 6,711 closed deals. Open deals are scored, but we never label them as lost.

**Source:** `reports/crm/final/data/data_quality.json`

---

## Slide 4 — Time-based split: no peeking at the future

**Layout:** Timeline with three proportional blocks (Train · Validation · Test) along the engage-date axis; big number "1,792" on the left; bullets on the right.

**On slide:**
- Train: 2,975 deals — engaged Oct 2016 – Jul 2017
- Validation: 583 deals — Jul – Sep 2017
- Test: 1,361 deals — engaged Sep – Dec 2017
- **1,792** closed deals purged: their outcome was not yet known at the cutoff
- Deals engaged on the same date always stay in the same period
- Train and validation outcomes must be known before the next period starts
- Imputation, encoding and scaling are fit on training rows only
- Check: a random split scores at most 0.04 AUC higher, still near chance

**Notes:** We split the data by time, not at random. Train has 2,975 deals, validation 583, and test 1,361. We removed 1,792 deals whose outcome was not yet known at the cutoff. So the model never learns from the future. We also checked a random split. It is at most 0.04 higher in ROC-AUC, and it is still near chance.

**Source:** `reports/crm/final/run_manifest.json` (split), `reports/crm/final/checks/split_protocol_summary.csv`

---

## Slide 5 — Six models, two feature sets, one pipeline

**Layout:** Six model cards in a row; below, a four-step process flow with icons.

**On slide:**
- Dummy prior — Control: always predicts the training win rate
- Logistic Regression — Linear baseline
- Random Forest — 300 trees, min leaf 5
- MLP — Hidden layers 64 → 32, early stopping
- TabNet — Attentive deep model for tabular data
- CatBoost — Gradient-boosted trees, native categories
- Two feature sets: **raw** (18 inputs) and **history** (+ prior win rates of agent, account and product, using only deals closed before engagement) → 12 candidates
- Process: **Select feature set + model** by validation ROC-AUC (locked before test) → **Pick threshold** maximise validation macro-F1 → **Calibrate** sigmoid on validation; base model frozen → **Test once** exploratory holdout
- Every model sees the same rows and the same preprocessing, trained on CPU with seed 42.

**Notes:** We trained six models: a Dummy control, Logistic Regression, Random Forest, MLP, TabNet and CatBoost. Each model is trained on two feature sets. The raw set has 18 inputs. The history set adds past win rates of the agent, the account and the product. These use only deals that closed before the new deal started. That is twelve candidates. We picked one using validation data only, and we locked it before looking at the test data. The winner was Logistic Regression with history features, with a validation ROC-AUC of 0.569.

**Source:** `reports/crm/final/models/selection_lock.json`, `reports/crm/final/metrics/feature_set_comparison.csv`

---

## Slide 6 — Why simulate? Original vs simulated data

**Layout:** Two columns (Original · Simulated) with one comparison table below.

**On slide:**
- **Original data: little to learn from**
  - Win rate barely changes across groups (products 0.60–0.65)
  - No deal-level signals: no discount, contacts, budget or competitor
  - Only 2,975 training deals after the time-based split
  - 68% of open deals have no account; no closed deal is missing one
- **Simulated data: signal we control**
  - A planted win formula, so the true win probability is known (oracle)
  - Deal-level features: sector match, industry trend, competitor
  - 100,000 deals, same format, same 85 accounts and dates
  - Signal strength and interaction type can be adjusted

| | Original | Simulated |
|---|---|---|
| Win rate across products | 0.60 – 0.65 | 0.27 – 0.93 |
| Best single-feature AUC | 0.56 | 0.76 |
| Closed deals | 6,711 | 71,463 |
| Best possible AUC (oracle) | unknown | 0.877 |

**Notes:** The real data has very little signal. The win rate is almost the same for every product and every sector. There is no information about calls, emails, budget or competitors. So we also made simulated data with a signal that we wrote ourselves. Then we know the true answer and the best possible score.

**Source:** `data/crm/` and `data/crm_simulated/` (win rates by group and single-feature AUC computed on all closed deals, so they are in-sample upper bounds); oracle from `data/crm_simulated/ground_truth.csv`

---

## Slide 7 — How we simulated 100,000 deals

**Layout:** Three columns with icons; a full-width "oracle" box below.

**On slide:**
- **Reused from real data:** 85 accounts, 30 agents, 7 products · Date range and 63% win rate · Share of open and Prospecting deals
- **Added by the team:** 9 new products, new GTS series · Competitor products · 5-year industry trends
- **Planted win formula:** Known logistic model · 3 interactions, e.g. expensive product × small account · Strength k = 3
- We know every deal's true win probability. Scoring with it gives the oracle: the best ROC-AUC any model can reach. It is never used as a feature.

**Notes:** The simulated data reuses the real accounts, agents and products. We added new products, competitors and industry trends. Each deal's win chance comes from a formula, including interactions. Because we wrote the formula, we know the best possible score. We call it the oracle. It is never used as a feature. Now my teammate will show the results.

**Source:** `SIMULATED_DATA.md`

---

## Slide 8 — Result A: no model beats chance

**Layout:** Full-width metrics table on top; three big-number callouts below.

**Table — test set (1,361 deals); selected feature set (history):**

| Model | Accuracy | Precision | Recall | F1 (Won) | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Dummy prior | 0.600 | 0.600 | 1.000 | 0.750 | 0.500 | 0.2421 |
| Logistic Regression (selected) | 0.426 | 0.608 | 0.124 | 0.206 | 0.517 | 0.3180 |
| Random Forest | 0.501 | 0.591 | 0.547 | 0.568 | 0.480 | 0.2537 |
| MLP | 0.516 | 0.618 | 0.506 | 0.556 | 0.499 | 0.2863 |
| TabNet | 0.525 | 0.599 | 0.633 | 0.616 | 0.502 | 0.2411 |
| CatBoost | 0.526 | 0.596 | 0.652 | 0.623 | 0.499 | 0.2443 |

**On slide:**
- **0.5168** — test ROC-AUC of the selected model (history LR); all 12 learned candidates land between 0.480 and 0.542 (0.5 = random ranking); raw LR 0.5238; raw CatBoost 0.542 is highest on test but cannot be chosen after the fact
- **60%** — accuracy of the Dummy, just by always predicting Won
- **0.3180 → 0.2489** — LR Brier after sigmoid calibration, still worse than the Dummy's 0.2421; calibration fixes the probability scale, not the ranking

**Notes:** Here is the main result on the real data. All twelve learned models have a test ROC-AUC between 0.480 and 0.542. A value of 0.5 means random ranking. The selected model has 0.5168. One model, CatBoost on raw features, looks a little better on test. But we do not change our choice after looking at the test data. That would be cheating. The Dummy gets 60 percent accuracy just by always saying "Won". So accuracy alone is misleading.

**Source:** `reports/crm/final/metrics/test_metrics.csv`, `reports/crm/final/metrics/feature_set_comparison.csv`, `reports/crm/final/metrics/calibration_test.csv`

---

## Slide 9 — Result B: nonlinear models near the oracle

**Layout:** Bar chart on the left (one bar per model, plus the oracle as a grey reference bar); big number and a five-seed card on the right.

**Chart — Test ROC-AUC on simulated data (Dummy 0.500):** Logistic Regression 0.809 · Random Forest 0.864 · MLP 0.866 · TabNet 0.869 · Oracle 0.878

**On slide:**
- **0.869** TabNet, vs oracle 0.878
- Logistic Regression trails because it cannot represent the planted interactions.
- Five regenerated datasets (k = 3): TabNet 0.875 ± 0.008 · MLP 0.873 ± 0.008 · Random Forest 0.872 ± 0.007 · Logistic Regression 0.827 ± 0.008 · Oracle 0.886 ± 0.007

**Notes:** On the simulated data, the same pipeline works well. TabNet reaches 0.869, and the best possible score is 0.878. So our pipeline can learn when there is a real signal.

**Source:** `reports/crm/simulated/final/metrics/test_metrics.csv` (main pipeline, `src.train --data-dir data/crm_simulated`; git-ignored), oracle from `data/crm_simulated/model_results.csv`, five seeds from `data/crm_simulated/seed_summary.csv`

---

## Slide 10 — Same pipeline, two datasets

**Layout:** Two large side-by-side cards (A vs B) with a big number each; conclusion line with a check icon below.

**On slide:**
- **A · Original CRM data — 0.517** test ROC-AUC of the selected model (all models 0.480–0.542): about the same as random. The available features carry almost no signal about Won or Lost.
- **B · Simulated data — 0.869** best test ROC-AUC (oracle 0.878). The same pipeline and models do learn when structure exists.
- The low scores come from the data, not from the pipeline. Simulated scores are never real-world performance and are kept out of the reported results.

**Notes:** Let us put the two results together. On the real data, every model is near chance. On the simulated data with a signal, the same code reaches about 0.87. So the low real score comes from the data, not from a broken pipeline. We never mix the simulated numbers into our real results.

---

## Slide 11 — Scoring open deals, and why each score

**Layout:** Left column: three stats. Right: one example deal card with "raises" and "lowers" lists. Below the card: a check row and a warning row, each with an icon.

**On slide:**
- **1,589** Engaging deals scored
- **1,088** flagged: account missing, so the score is less reliable
- **712 / 876 / 1** High / Medium / Low priority (heuristic bands 0.70 and 0.40)
- Example — Deal 125VIRMX · Warephase · GTK 500 · Elease Gluck · **98%** P(Won) · High priority
  - Raises the score: engage_year = 2016: +0.139 · sales_price = 26768: +0.079 · global_win_rate = 0.5: +0.075
  - Lowers the score: global_mean_cycle_days = 0: −0.023 · product_cold_start = 1: −0.015 · engage_quarter = 4: −0.015
  - Values are changes in P(Won) when that input is reset to the training reference.
- ✓ Checked: factors agree with RF TreeSHAP (median Spearman 0.54; top-3 overlap 73% vs 8% by chance); resetting the top 3 moves P(Won) 6.3× more than random. Associations, not causes.
- ⚠ Priority bands are a heuristic, not a validated sales policy. Scores come from a frozen model snapshot, not historical as-of predictions.

**Notes:** We scored 1,589 open deals. But 1,088 of them have no account information. That is 68 percent. The model never saw deals like this in training. For each deal we also show the factors that move the score. These are associations, not causes.

**Source:** `reports/crm/final/predictions/open_deal_predictions.csv`, `reports/crm/final/checks/explanation_checks.json`

---

## Slide 12 — Where traditional ML stops

**Layout:** Left: a "score card" mock-up of the traditional output (one number, label, factors, "Warning: none"). Right: three big-number callouts for the hidden problems. Bottom: one-line takeaway.

**On slide:**
- Traditional output for deal 01XZ9CRY: P(Won) 0.534 · predicted Lost · priority Medium · top factor global_closed_count −0.246 · Warning: none
- **68%** of open deals (1,088 of 1,589) have no account: outside training support
- **52%** of open deals (827) have at least one input outside the training range
- **0.569** validation ROC-AUC (95% CI 0.523–0.616): a weak model
- A score alone does not say whether to trust it, what evidence supports it, or what to do next.

**Notes:** A traditional pipeline stops here. It gives a number and some factors. Then a person must check everything by hand. And there are hidden problems. In 827 of the 1,589 open deals, at least one input is outside the range the model saw in training. The pipeline gives no warning for that. With a weak model, missing data and hidden extrapolation, a single score is not enough to decide what to do.

**Source:** `python -m src.agent_demo --traditional`; `reports/crm/final/agent/open_decisions.csv` (flags); `reports/crm/final/models/model_card.json`

---

## Slide 13 — Agent architecture

**Layout:** Full-width architecture diagram (`docs/figures/agent_architecture.png`); one caption line below.

**On slide:**
- Diagram: frozen ML pipeline + read-only CRM data → agent controller (goal → observe → plan → act → evaluate → decide, max 14 steps) → local tools / local writes / external actions blocked
- Caption: Deterministic and offline. No LLM. Plan = need − cost; evidence-tool need = value of information. Every step goes into a replayable trace.

**Notes:** So we built an agent on top of the frozen model. It works in a loop. First it sets a goal. Then it observes the deal. Then it plans: every tool gets a score, need minus cost. It calls the best tool, and it evaluates the result. When no tool is worth calling, it decides. It can only read data, compute, and write local files. It can never email a customer or change the CRM. Every step goes into a trace.

**Source:** `docs/AGENT_ARCHITECTURE.md`, `src/agent/`

---

## Slide 14 — What makes it different

**Layout:** Comparison table, three system columns; the agent column highlighted.

**Table:**

| | A · Prediction only | B · Rule assistant | C · Our agent |
|---|---|---|---|
| Chooses its own steps | No | No: fixed rules, fixed order | Yes: need − cost, value of information |
| Checks data quality and model reliability | Missing account only | Missing account, timing | Missing, imputed, out-of-range, stale, committee, validation CI |
| Refuses unsupported labels | No | Partly | Yes: evidence-gated decision |
| Executes actions | No | No (prints advice) | Local review tasks only |
| External actions | — | — | Blocked; queued for human approval |
| Trace / replay | No | No | Yes: content hash, tool replay |

**Notes:** This table compares the three systems on the same deals. The traditional pipeline gives a score. The rule-based assistant adds fixed rules, in the same order every time. Our agent chooses its own tools, based on whether the evidence could change the decision. It refuses to give a confident label when the evidence is weak. It blocks unsafe actions. And it records everything, so the decision can be replayed. Now my teammate will show it running.

**Source:** `src/agent/`, `src/agent/assistant.py`, `src/explain/reference.py`

---

## Slide 15 — The agent on one real deal

**Layout:** Vertical step list (the trace of deal 01XZ9CRY) on the left; decision card on the right.

**On slide:**
- Steps: goal → get_opportunity → get_account_information → check_data_quality (warning: outside_training_range) → predict_win_probability (0.534 < threshold 0.611: at risk) → get_prior_sales_history → explain_prediction → check_model_agreement → evaluate_evidence → decide → create_review_task
- Decision: **ESCALATE_AT_RISK_REVIEW** — high-priority local review task; 12 steps; stopped by itself; 0 external actions
- Why: explanation and model agreement ran because their result could change the action (value of information); prior history ran as context for the review task

**Notes:** Here is one real deal, 01XZ9CRY. The agent looked up the record and the account. It checked the data quality and found inputs outside the training range. It got the score, which is below the threshold. It checked the history and the explanation, and it asked the other models if they agree. The evidence was strong enough, so it created a high-priority review task. It took twelve steps, and it stopped by itself.

**Source:** `reports/crm/final/agent/scenario_runs/S1/traces/01XZ9CRY.full_agent.report.md`

---

## Slide 16 — Live demo

**Layout:** Section divider (dark background), with the one command shown.

**On slide:**
- Live demo
- `python -m src.agent_demo`

**Notes:** Now I will run it live. Everything is already trained, so this takes a few seconds. (One person operates the demo, following the 3–5 minute table in `docs/EXCELLENT_PROJECT_DEMO_STEP_BY_STEP.md`.)

**Source:** `src/agent_demo.py`; fallback: saved traces in `reports/crm/final/agent/scenario_runs/`

---

## Slide 17 — Baseline vs agent on 1,589 real open deals

**Layout:** Full-width figure `reports/crm/final/figures/agent_comparison.png`; three callouts below.

**On slide:**
- **1,242 → 487 → 0** unsupported directional labels (A → B → agent)
- **0.28 → 0.64 → 1.00** recall of known data issues
- **5.0 → 3.0 → 2.2** manual steps left (of 9); 100% of the agent's claims re-verified; +27 ms per deal vs B
- Same frozen model, same deals. Workflow properties, not sales outcomes.

**Notes:** We compared the three systems on the same 1,589 real open deals and the same frozen model. The traditional pipeline gave a directional label to every out-of-support or borderline deal. That is 1,242 unsupported labels. The rule assistant gave 487. Our agent gave zero. The agent found all known data problems, while the rule assistant found 64 percent of them. Every one of the agent's evidence claims was checked again from the raw files, and all of them matched. The cost is small: about 79 milliseconds per deal, compared with 52 for the rule assistant.

**Source:** `reports/crm/final/agent/system_comparison.csv`, `reports/crm/final/agent/scenario_results.csv`

---

## Slide 18 — Ablations and an honest offline check

**Layout:** Left: figure `reports/crm/final/figures/agent_ablation.png`. Right: replay table.

**On slide:**
- Without reliability checks: **80%** of decisions change; unsupported labels **53%**
- Without planning: same decisions, **+2.1** tool calls and **2.9×** runtime per deal
- Without evidence retrieval: **10%** of decisions change
- Outcome-masked replay on 1,361 test deals (base win rate 60.0%): escalated deals won **62.3%**: no better than chance at finding losses

**Notes:** Then we turned off one part at a time. Without the reliability checks, 80 percent of the decisions change, and unsupported labels come back. Without planning, the decisions stay the same, but the agent calls about two more tools per deal and is almost three times slower. So planning saves cost. We also replayed the agent on 1,361 closed test deals with the outcome hidden. The deals it flagged as at risk were won 62 percent of the time, and the base rate is 60 percent. So the agent does not find losing deals better than chance. That is honest: it cannot be better than the model.

**Source:** `reports/crm/final/agent/ablation_summary.csv`, `reports/crm/final/agent/replay_outcomes.csv`

---

## Slide 19 — What our project adds

**Layout:** Contribution table (contribution · implementation · evidence), four rows; one "not claimed" line below.

**Table:**

| Contribution | Implementation | Evidence |
|---|---|---|
| Value-of-information tool planning | `src/agent/planner.py` | Same decisions with 2.1 fewer tool calls per deal |
| Reliability-gated decisions | `planner.decision_candidates` | 0 unsupported labels vs 1,242 (A), 487 (B) |
| Bounded autonomy + approval queue | `src/agent/policy.py` | 0 external actions; prohibited request blocked |
| Replayable, source-cited traces | `src/agent/controller.py` | 200/200 identical hashes; 880/880 tool replays |

- Not claimed: a new learning algorithm, better ROC-AUC, online learning, LLM use, or sales uplift.

**Notes:** Our novelty is not a new learning algorithm. It is the agent architecture. One: it plans which evidence to collect, using value of information. Two: it gates its decisions on evidence quality and model reliability. Three: it acts on its own only inside safe, local limits. Four: every decision has a trace that we can replay. Each point is measured against two baselines and with ablations.

**Source:** `NOVELTY_AND_RESULTS.md`, `docs/AGENT_ARCHITECTURE.md`

---

## Slide 20 — Limitations and future work

**Layout:** Two columns (Limitations · Future work); takeaway line at the bottom.

**On slide:**
- **Limitations**
  - Trained on closed deals only: selection bias
  - Validation reused for model, threshold and calibration
  - Account, team and product tables are static snapshots
  - Accounts repeat across periods
  - Data was examined earlier, so the test set is not pristine
  - Simulated win rule is invented: says nothing about real sales
  - Agent thresholds are design choices, not learned from feedback
  - No sales-team test: no measured uplift; offline replay shows no outcome-targeting gain
- **Future work**
  - Collect engagement signals: calls, emails, response time, budget
  - Score deals as of each historical date
  - Test on accounts never seen in training
  - Pilot the agent's review queue with a sales team and collect feedback
  - Learn the agent's thresholds from real feedback (only then could it learn online)
  - Train-on-synthetic, test-on-real check with a data-driven generator
- Takeaway: a rigorous, time-aware pipeline and a safe, auditable agent; the current CRM fields are not enough to rank deals.

**Notes:** There are clear limits. The real data has little signal, so the prediction is weak. The agent's thresholds are design choices, not learned from feedback. We did not measure any increase in sales. In the future, we need engagement data like calls and emails, a test with a real sales team, and real feedback so the agent could learn.

---

## Slide 21 — Thank you — questions?

**Layout:** Closing slide (dark background).

**On slide:**
- Thank you — questions?
- Hong Thai Phan · Nguyen Khanh An Tran · Hoang Thien Bao Bui
- References: J. Yan et al., Sales Pipeline Win Propensity Prediction, IFIP/IEEE IM 2015 · A. Rezazadeh, A Generalized Flow for B2B Sales Predictive Modeling, Forecasting 2020 · S. O. Arik and T. Pfister, TabNet, AAAI 2021 · S. Russell and P. Norvig, AIMA, 4th ed. 2020 · R. A. Howard, Information Value Theory, 1966

**Notes:** Thank you for listening. We are happy to answer your questions.

---

## Backup B1 — How MLP and TabNet work

**Layout:** Two columns, one per model, each with a small flow diagram on top (MLP: stacked dense layers; TabNet: three decision steps, each with a feature mask). A comparison table spans the bottom.

**On slide:**
- **MLP (multilayer perceptron)**
  - Fully connected layers: every neuron sees every input at once
  - Our network: inputs → 64 ReLU → 32 ReLU → sigmoid P(Won)
  - Feature interactions are learned implicitly inside weighted sums
  - Regularised by L2 weight decay and early stopping
  - No built-in view of which inputs matter: needs post-hoc methods (permutation, SHAP)
- **TabNet (Arik & Pfister, 2021)**
  - Several sequential decision steps (we use 3)
  - At each step an attentive transformer outputs a sparse mask (sparsemax) that selects a few inputs, separately for every deal
  - A feature transformer (shared + step-specific GLU blocks) processes only the selected inputs; step outputs are added up
  - A prior term discourages reusing the same input in every step; a sparsity loss keeps masks small
  - The masks are a built-in, per-deal explanation of which inputs were used

| | MLP | TabNet |
|---|---|---|
| Which inputs are used | All inputs, in every layer | A learned subset per step, per deal |
| How it decides | One dense, smooth function | Sequential, tree-like feature selection |
| Explanation | Post-hoc only | Attention masks built in |
| Hyperparameters | Few (layers, width, decay) | More (steps, widths, γ, sparsity, virtual batch) |
| Training cost | Light | Heavier: attention and several steps per update |

**Notes:** Before comparing results, here is how the two deep models differ. An MLP is a stack of fully connected layers. Every neuron looks at every input, and the network learns interactions inside its weighted sums. Ours has two hidden layers, 64 and 32 units. It has no built-in idea of which inputs matter, so we explain it with methods like permutation importance or SHAP. TabNet was designed for tabular data. It works in several steps. At each step, an attention module chooses a small set of inputs for that particular deal, using a sparse mask. Only those inputs are processed in that step, and the results of all steps are added together. This makes TabNet behave a bit like a sequence of decision-tree splits, and the masks show which inputs it used for each deal. The cost is more hyperparameters and slower training.

**Source:** Arik and Pfister, TabNet, AAAI 2021; model settings in `src/models/zoo.py`

---

## Backup B2 — TabNet vs MLP: no clear accuracy winner

**Layout:** Comparison table on the left; two big-number callouts on the right; takeaway line below.

**Table:**

| Comparison | MLP | TabNet |
|---|---|---|
| Real data (history set): test ROC-AUC | 0.499 | 0.502 |
| Real data (history set): validation ROC-AUC | 0.535 | 0.504 |
| Simulated, interactions (5 seeds) | 0.873 | 0.875 |
| Simulated, smooth signals (5 seeds) | 0.866 | 0.868 |
| Trainable parameters | 7.6k | 11.3k |
| Train time: real / 100k simulated | 0.9 s / 3 s | 6.2 s / 107 s |
| Built-in explanation | None | Attention masks |

**On slide:**
- **+0.003** TabNet − MLP test ROC-AUC on real data; 95% CI −0.045 to +0.051, so not a real win
- **7×** slower for TabNet to train on real data (35× on 100k simulated deals)
- Accuracy differences sit inside the noise in every setting. TabNet adds built-in feature attention, at a much higher training cost.

**Notes:** Both MLP and TabNet are neural networks, so this is the fairest deep-learning comparison. On the real data TabNet scores 0.502 and the MLP 0.499. The 95 percent confidence interval for the difference goes from minus 0.045 to plus 0.051, so it is not a real win. On simulated data they are within 0.002 AUC: TabNet is ahead in four of five seeds with interactions, and three of five with smooth signals. TabNet is about 7 times slower to train here, and 35 times slower on 100,000 deals. What TabNet does give us is built-in attention, which shows the features each decision step used. The MLP needs a separate method for that.

**Source:** `reports/crm/final/metrics/` (test/validation, `test_predictions.csv` for the paired bootstrap), `reports/crm/final/models/model_training.json` (train time), `data/crm_simulated/seed_results*.csv`

---

## Backup B3 — Scenario B: does the pipeline learn when signal exists?

**Layout:** Section divider (dark background, title in the middle).

**On slide:**
- Title: Scenario B: does the pipeline learn when signal exists?
- Subtitle: Same loader, split, preprocessing, threshold rule and metrics on 100,000 simulated deals

**Notes:** Low scores could mean the data has no signal, or that our pipeline is broken. To tell these apart, we ran the same pipeline on simulated data where we know the signal exists.
