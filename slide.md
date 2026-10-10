# CRM_Final — slide content for review

Cách review: mỗi slide có **Layout** (bố cục / hình), **On slide** (chữ hiện trên slide), **Notes** (speaker notes) và **Source** (file chứa con số). Sửa trực tiếp file này; khi bạn duyệt xong mình mới tạo `CRM_Final.pptx` từ đây.

Nội dung dựa trên `CRM_Final.pptx` hiện tại (19 slide), cộng 3 slide mới, bớt 4 slide (18 slide). Mọi con số đã được đối chiếu với `reports/crm/final/` (lần chạy macOS ngày 2026-10-09) và `data/crm_simulated/`. Thay đổi so với deck hiện tại:
- Thứ tự mới: phương pháp (slide 1–9) → **slide 10 "Demo" (mới)** → kết quả (slide 11–15) → kết luận (16–18).
- Slide 6 (mới): phân tích lý thuyết MLP và TabNet hoạt động khác nhau thế nào (đặt ngay sau slide 5 model).
- Slide 7 (mới): vì sao cần dữ liệu mô phỏng — nhược điểm dữ liệu gốc và ưu điểm dữ liệu mô phỏng.
- Slide 11: gộp "Result A" và "Full test metrics and calibration" thành một slide; khoảng AUC ghi rõ 0.495–0.524.
- Bỏ slide "Leakage control: hide the answer" và slide "3-month expected revenue (illustrative)"; bỏ mọi câu nhắc tới close_value/leakage ở các slide khác (slide 16 còn 5 thẻ).
- Slide 12: gộp "What moves the score? Associations only" vào "Scoring the open pipeline" (bỏ biểu đồ permutation importance, giữ phần kiểm tra giải thích).
- Slide 13 (Result B): bỏ so sánh base vs extended features; mọi số liệu mô phỏng dùng extended features (mặc định).
- Slide 13, 15, 16: TabNet trên dữ liệu mô phỏng đổi từ 0.867 sang 0.869, theo kết quả của pipeline chính (cũng là số notebook simulated sẽ hiện khi demo).
- Slide 16 (What our project adds): thẻ cuối nhắc tới notebook từng bước.

---

## Slide 1 — Title

**Layout:** Title slide.

**On slide:**
- Title: Predicting CRM Sales Opportunities Using Machine Learning
- Subtitle: CS 582 Machine Learning · Group 2 · Final presentation
- Hong Thai Phan · Nguyen Khanh An Tran · Hoang Thien Bao Bui

**Notes:** Hello everyone. We are Group 2. Our project is Predicting CRM Sales Opportunities Using Machine Learning. We predict whether a sales opportunity will be Won or Lost, and we show which factors influence the score.

---

## Slide 2 — Goal: a win probability and its reasons

**Layout:** Three cards in a row (Input → Models → Output), then a two-line box for the two scenarios.

**On slide:**
- **Input:** Four CRM tables: sales pipeline, accounts, products, sales teams
- **Models:** Dummy control, Logistic Regression, Random Forest, MLP, TabNet
- **Output:** P(Won) for each open deal plus the factors that move its score
- **Two scenarios, same pipeline**
  - A · Original Maven CRM data: the project's reported results.
  - B · Simulated data with a known formula: a check that the pipeline finds signal when signal exists.

**Notes:** The input is four CRM tables: pipeline, accounts, products and sales teams. We train five models, including a Dummy control. The output is a win probability for each open deal, plus the factors that move the score. We show two scenarios. Scenario A uses the original Maven CRM data and gives our reported results. Scenario B runs the same models on simulated data, to check that the pipeline finds signal when signal exists.

---

## Slide 3 — Data: a fictitious B2B hardware seller

**Layout:** Bar chart on the left (deal stage of all 8,800 opportunities); four stat callouts on the right.

**Chart — Deal stage of all 8,800 opportunities:** Won 4,238 · Lost 2,473 · Engaging (open) 1,589 · Prospecting (open) 500

**On slide:**
- 8,800 sales opportunities · 85 customer accounts · 7 products · 35 sales agents
- Train only on 6,711 closed deals (63% Won). Open deals are scored, never labelled Lost.
- Source: Maven Analytics CRM Sales Opportunities (public practice dataset)

**Notes:** The data is the Maven Analytics CRM dataset: a fictitious company that sells computer hardware to businesses. It has 8,800 opportunities, 85 accounts, 7 products and 35 sales agents. We join the four tables and fix two spelling problems in the raw data. We train only on the 6,711 closed deals: 4,238 Won and 2,473 Lost. Open deals are scored, never treated as Lost.

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

**Notes:** We split by time, not at random. Train has 2,975 deals, validation 583 and test 1,361. Deals from the same date stay together. We purged 1,792 rows whose outcomes were not known at the cutoff, so the model never learns from the future. Preprocessing is fit on the training set only. We also checked whether this matters: a random split gives at most 0.04 higher AUC, and every protocol is still near chance. So the split is a safeguard, not the reason our scores are low.

**Source:** `reports/crm/final/run_manifest.json` (split), `reports/crm/final/checks/split_protocol_summary.csv`

---

## Slide 5 — Five models, one shared pipeline

**Layout:** Five model cards in a row; below, a four-step process flow with icons.

**On slide:**
- Dummy prior — Control: always predicts the training win rate
- Logistic Regression — Linear baseline
- Random Forest — 300 trees, min leaf 5
- MLP — Hidden layers 64 → 32, early stopping
- TabNet — Attentive deep model for tabular data
- Process: **Select model** by validation ROC-AUC → **Pick threshold** maximise validation macro-F1 → **Calibrate** sigmoid on validation; base model frozen → **Test once** exploratory holdout
- Every model sees the same rows and the same preprocessing, trained on CPU with seed 42.

**Notes:** We trained a Dummy control, Logistic Regression, Random Forest, MLP and TabNet on the same rows and the same preprocessing. The model, the threshold and the calibration are all chosen on validation. Logistic Regression had the best validation ROC-AUC, 0.563, so it was selected before we looked at test. The threshold maximises macro-F1 on validation instead of using 0.5.

**Source:** `reports/crm/final/metrics/validation_metrics.csv`

---

## Slide 6 — How MLP and TabNet work

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

## Slide 7 — Why simulate? Original vs simulated data

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

**Notes:** The original data has almost no signal. The win rate is nearly the same for every product, sector and agent, and the file has no deal-level information such as discounts or contacts. The simulated data keeps the same format but adds a signal we wrote ourselves, so we know the true answer and the best possible score. It only tests the pipeline; it says nothing about real sales.

**Source:** `data/crm/` and `data/crm_simulated/` (win rates by group and single-feature AUC computed on all closed deals, so they are in-sample upper bounds); oracle from `data/crm_simulated/ground_truth.csv`

---

## Slide 8 — Scenario B: does the pipeline learn when signal exists?

**Layout:** Section divider (dark background, title in the middle).

**On slide:**
- Title: Scenario B: does the pipeline learn when signal exists?
- Subtitle: Same loader, split, preprocessing, threshold rule and metrics on 100,000 simulated deals

**Notes:** Low scores could mean the data has no signal, or that our pipeline is broken. To tell these apart, we ran the same pipeline on simulated data where we know the signal exists.

---

## Slide 9 — How we simulated 100,000 deals

**Layout:** Three columns with icons; a full-width "oracle" box below.

**On slide:**
- **Reused from real data:** 85 accounts, 30 agents, 7 products · Date range and 63% win rate · Share of open and Prospecting deals
- **Added by the team:** 9 new products, new GTS series · Competitor products · 5-year industry trends
- **Planted win formula:** Known logistic model · 3 interactions, e.g. expensive product × small account · Strength k = 3
- We know every deal's true win probability. Scoring with it gives the oracle: the best ROC-AUC any model can reach. It is never used as a feature.

**Notes:** The simulated data reuses the real accounts, agents and products, and matches the real date range and win rate. We added new products, competitor products and industry trends. Each deal's win probability comes from a formula we wrote, including interaction effects such as an expensive product sold to a small account. Because we wrote the formula, we know the true probability, which gives us the oracle: the best possible score. It is never used as a feature.

**Source:** `SIMULATED_DATA.md`

---

## Slide 10 — Demo

**Layout:** Section divider (dark background, the word in the middle).

**On slide:**
- Demo

**Notes:** Now a short live demo. We run the three notebooks for the original data in order: 1_train fits the five models and selects one on validation, 2_evaluate computes the test metrics and figures from the saved models, and 3_predict scores the open deals and shows the factors behind one score. If time allows, we show the same three steps on the simulated data. After the demo we go through the results.

**Source:** `notebooks/original/`, `notebooks/simulated/`; walkthrough in `demo_step_by_step.md`

---

## Slide 11 — Result A: no model beats chance

**Layout:** Full-width metrics table on top; three big-number callouts below.

**Table — test set (1,361 deals):**

| Model | Accuracy | Precision | Recall | F1 (Won) | ROC-AUC | Brier |
|---|---|---|---|---|---|---|
| Dummy prior | 0.600 | 0.600 | 1.000 | 0.750 | 0.500 | 0.2421 |
| Logistic Regression (selected) | 0.437 | 0.615 | 0.166 | 0.262 | 0.524 | 0.2794 |
| Random Forest | 0.592 | 0.598 | 0.978 | 0.742 | 0.518 | 0.2510 |
| MLP | 0.485 | 0.593 | 0.453 | 0.514 | 0.495 | 0.3314 |
| TabNet | 0.536 | 0.600 | 0.678 | 0.637 | 0.507 | 0.2490 |

**On slide:**
- **0.524** — test ROC-AUC of the selected model; all learned models land between 0.495 and 0.524 (0.5 = random ranking)
- **60%** — accuracy of the Dummy, just by always predicting Won
- **0.2794 → 0.2443** — LR Brier after sigmoid calibration, still worse than the Dummy's 0.2421; calibration fixes the probability scale, not the ranking

**Notes:** Here is the main result on the original data. Test ROC-AUC is between 0.495 and 0.524 for all models, and 0.5 means random ranking. The selected Logistic Regression has 0.524. Its recall is low because its threshold, chosen on validation, is strict. The Dummy gets 60 percent accuracy only because about 60 percent of deals are Won and it always predicts Won, so accuracy alone is misleading. Calibration improved the Brier score of Logistic Regression from 0.2794 to 0.2443, but that is still slightly worse than the Dummy, and it does not change the ranking.

**Source:** `reports/crm/final/metrics/test_metrics.csv`, `reports/crm/final/metrics/calibration_test.csv`

---

## Slide 12 — Scoring open deals, and why each score

**Layout:** Left column: three stats. Right: one example deal card with "raises" and "lowers" lists. Below the card: a check row and a warning row, each with an icon.

**On slide:**
- **1,589** Engaging deals scored
- **1,088** flagged: account missing, so the score is less reliable
- **400 / 1,189 / 0** High / Medium / Low priority (heuristic bands 0.70 and 0.40)
- Example — Deal AI76U58A · Treequote · GTX Basic · Maureen Marcano · **88%** P(Won) · High priority
  - Raises the score: engage_year = 2016: +0.294 · sales_agent = Maureen Marcano: +0.029 · engage_month = 12: +0.020
  - Lowers the score: engage_quarter = 4: −0.064 · sector = telecommunications: −0.012 · regional_office = West: −0.002
  - Values are changes in P(Won) when that input is reset to the training reference.
- ✓ Checked: factors agree with RF TreeSHAP (median Spearman 0.76); resetting the top 3 moves P(Won) 3.1× more than random. Associations, not causes.
- ⚠ Priority bands are a heuristic, not a validated sales policy. Scores come from a frozen model snapshot, not historical as-of predictions.

**Notes:** We scored 1,589 Engaging deals. Win and loss probabilities sum to one. 1,088 have a missing account and are flagged. Each deal gets a priority band and the factors that raise or lower its score: we reset one input at a time to its training median or mode and measure the change in win probability. This example is the highest-scored deal. We checked that these factors describe the model: on the Random Forest they agree with TreeSHAP, with a median Spearman correlation of 0.76, and resetting a deal's top three factors moves the score 3.1 times more than resetting three random ones. They show what the model reacts to, not causes, and the priority bands are a heuristic that has not been validated.

**Source:** `reports/crm/final/predictions/open_deal_predictions.csv`, `reports/crm/final/checks/explanation_checks.json`

---

## Slide 13 — Result B: nonlinear models near the oracle

**Layout:** Bar chart on the left (one bar per model, plus the oracle as a grey reference bar); big number and a five-seed card on the right.

**Chart — Test ROC-AUC on simulated data (Dummy 0.500):** Logistic Regression 0.809 · Random Forest 0.864 · MLP 0.866 · TabNet 0.869 · Oracle 0.878

**On slide:**
- **0.869** TabNet, vs oracle 0.878
- Logistic Regression trails because it cannot represent the planted interactions.
- Five regenerated datasets (k = 3): TabNet 0.875 ± 0.008 · MLP 0.873 ± 0.008 · Random Forest 0.872 ± 0.007 · Logistic Regression 0.827 ± 0.008 · Oracle 0.886 ± 0.007

**Notes:** On the simulated data, test ROC-AUC is 0.809 for Logistic Regression, 0.864 for Random Forest, 0.866 for MLP and 0.869 for TabNet. The best possible is 0.878. So the nonlinear models recover nearly all the planted signal. Over five newly generated datasets the result holds, with standard deviations under 0.01. Nonlinear models win partly because we built the data with interactions.

**Source:** `reports/crm/simulated/final/metrics/test_metrics.csv` (main pipeline, `src.train --data-dir data/crm_simulated`; git-ignored), oracle from `data/crm_simulated/model_results.csv`, five seeds from `data/crm_simulated/seed_summary.csv`

---

## Slide 14 — TabNet vs MLP: no clear accuracy winner

**Layout:** Comparison table on the left; two big-number callouts on the right; takeaway line below.

**Table:**

| Comparison | MLP | TabNet |
|---|---|---|
| Real data: test ROC-AUC | 0.495 | 0.507 |
| Real data: validation ROC-AUC | 0.529 | 0.515 |
| Simulated, interactions (5 seeds) | 0.873 | 0.875 |
| Simulated, smooth signals (5 seeds) | 0.866 | 0.868 |
| Trainable parameters | 7.6k | 11.3k |
| Train time: real / 100k simulated | 0.3 s / 3 s | 8 s / 107 s |
| Built-in explanation | None | Attention masks |

**On slide:**
- **+0.011** TabNet − MLP test ROC-AUC on real data; 95% CI −0.030 to +0.054, so not a real win
- **27×** slower for TabNet to train on real data (35× on 100k simulated deals)
- Accuracy differences sit inside the noise in every setting. TabNet adds built-in feature attention, at a much higher training cost.

**Notes:** Both MLP and TabNet are neural networks, so this is the fairest deep-learning comparison. On the real data TabNet scores 0.507 and the MLP 0.495. But the 95 percent confidence interval for the difference goes from minus 0.030 to plus 0.054, so it is not a real win. On simulated data they are within 0.002 AUC: TabNet is ahead in four of five seeds with interactions, and three of five with smooth signals. TabNet is about 27 times slower to train here, and 35 times slower on 100,000 deals. What TabNet does give us is built-in attention, which shows the features each decision step used. The MLP needs a separate method for that.

**Source:** `reports/crm/final/metrics/` (test/validation, `test_predictions.csv` for the paired bootstrap), `reports/crm/final/models/model_training.json` (train time), `data/crm_simulated/seed_results*.csv`

---

## Slide 15 — Same pipeline, two datasets

**Layout:** Two large side-by-side cards (A vs B) with a big number each; conclusion line with a check icon below.

**On slide:**
- **A · Original CRM data — 0.524** best test ROC-AUC: about the same as random. The available features carry almost no signal about Won or Lost.
- **B · Simulated data — 0.869** best test ROC-AUC (oracle 0.878). The same pipeline and models do learn when structure exists.
- The low scores come from the data, not from the pipeline. Simulated scores are never real-world performance and are kept out of the reported results.

**Notes:** Putting the two scenarios together: on the original data every model is near 0.52, and on simulated data with real signal the same pipeline reaches about 0.87. So the low scores come from the original data, not from the pipeline. The simulated scores are not real-world performance, and we never mix them into our reported results.

---

## Slide 16 — What our project adds

**Layout:** Five cards (three on top, two below), each with an icon, a title and one line of evidence.

**On slide:**
- **Prediction plus explanation** — P(Won) per deal; its top factors move the score 3.1× more than random ones
- **Label-availability split** — Train only on known outcomes; a random split adds at most 0.04 AUC
- **Input warnings** — 1,088 of 1,589 open-deal scores flagged for a missing account
- **Known-truth simulation** — Pipeline reaches 0.869 vs an oracle of 0.878 when signal exists
- **Reproducible on CPU** — Separate train, evaluate and predict steps, as commands or step-by-step notebooks

**Notes:** Our contribution is not a new algorithm. It is a careful workflow where each part has evidence. The explanations move the score three times more than random factors. A random split changes AUC by at most 0.04, so our split is a safeguard, not a trick. Missing accounts are flagged. And the simulation shows the pipeline learns when signal exists. Everything runs on a CPU, either with one command or step by step in notebooks.

---

## Slide 17 — Limitations and future work

**Layout:** Two columns (Limitations · Future work); takeaway line at the bottom.

**On slide:**
- **Limitations**
  - Trained on closed deals only: selection bias
  - Validation reused for model, threshold and calibration
  - Account, team and product tables are static snapshots
  - Accounts repeat across periods
  - Data was examined earlier, so the test set is not pristine
  - Simulated win rule is invented: says nothing about real sales
- **Future work**
  - Collect engagement signals: calls, emails, response time, budget
  - Score deals as of each historical date
  - Test on accounts never seen in training
  - Validate priority bands with the sales team
  - Train-on-synthetic, test-on-real check with a data-driven generator
- Takeaway: a rigorous, time-aware pipeline that works; the current CRM fields are not enough to rank deals.

**Notes:** Our results do not yet show the model helps sales prioritisation. Limitations: closed-only bias, validation reuse, static snapshots, repeated accounts, and the data was examined in earlier attempts, so the test set is not pristine. The simulated win rule is invented, so it says nothing about real sales. The most useful next step is richer engagement data, such as calls, emails and response times.

---

## Slide 18 — Thank you — questions?

**Layout:** Closing slide (dark background).

**On slide:**
- Thank you — questions?
- Hong Thai Phan · Nguyen Khanh An Tran · Hoang Thien Bao Bui
- References: J. Yan et al., Sales Pipeline Win Propensity Prediction, IFIP/IEEE IM 2015 · A. Rezazadeh, A Generalized Flow for B2B Sales Predictive Modeling, Forecasting 2020 · S. O. Arik and T. Pfister, TabNet, AAAI 2021

**Notes:** Thank you. We are happy to take questions. Likely questions: Why are the scores so low? The available features carry little signal. Why does the Dummy have high accuracy? About 60 percent of closed deals are Won. Is simulated data cheating? No: it only checks the pipeline, and our reported results use the original data.
