# Final presentation — ESL speaking transcript (three speakers)

Deck: `CRM_Final.pptx` (slide sources and notes: `slide.md`). Every number below comes from the verified run in `reports/crm/final/` and is checked by `tests/test_documents.py`.

- **Speaker 1** — slides 1–7: problem, data, split, models, simulation (≈ 5 min)
- **Speaker 2** — slides 8–14: ML results, why prediction is hard, the agent design (≈ 5 min)
- **Speaker 3 (demo operator)** — slides 15–21: workflow, **live demo**, experiments, novelty, conclusion (≈ 5 min + 3–4 min demo)

Team members decide who takes which speaker role. One person, Speaker 3, operates the whole live demo, following `docs/EXCELLENT_PROJECT_DEMO_STEP_BY_STEP.md`.

Style tips: speak slowly. Pause at each full stop. Point at the slide when you say a number.

---

## Speaker 1

**Slide 1 — Title.**
Hello everyone. We are Group 2. Our project is "Predicting CRM Sales Opportunities Using Machine Learning". We predict whether a sales deal will be won or lost. Then we built an agent that uses this prediction carefully and safely.

**Slide 2 — Goal.**
Our input is four CRM tables: the sales pipeline, accounts, products and sales teams. We train six models. The output is a win probability for each open deal and the factors behind it. On top of that, an agent collects evidence, checks it, and decides the next safe step. We use two data scenarios. Scenario A is the real CRM data, and these are our reported results. Scenario B is simulated data with a known formula. We only use it to check that our pipeline can learn when there is a signal.

**Slide 3 — Data.**
The data is the Maven Analytics CRM dataset. It is a fictitious company that sells computer hardware to other businesses. It has 8,800 deals, 85 accounts, 7 products and 35 sales agents. We train only on 6,711 closed deals. Open deals are scored, but we never label them as lost.

**Slide 4 — Time-based split.**
We split the data by time, not at random. Train has 2,975 deals, validation 583, and test 1,361. We removed 1,792 deals whose outcome was not yet known at the cutoff. So the model never learns from the future. We also checked a random split. It is at most 0.04 higher in ROC-AUC, and it is still near chance.

**Slide 5 — Six models, two feature sets.**
We trained six models: a Dummy control, Logistic Regression, Random Forest, MLP, TabNet and CatBoost. Each model is trained on two feature sets. The raw set has 18 inputs. The history set adds past win rates of the agent, the account and the product. These use only deals that closed before the new deal started. That is twelve candidates. We picked one using validation data only, and we locked it before looking at the test data. The winner was Logistic Regression with history features, with a validation ROC-AUC of 0.569.

**Slide 6 — Why simulate?**
The real data has very little signal. The win rate is almost the same for every product and every sector. There is no information about calls, emails, budget or competitors. So we also made simulated data with a signal that we wrote ourselves. Then we know the true answer and the best possible score.

**Slide 7 — How we simulated.**
The simulated data reuses the real accounts, agents and products. We added new products, competitors and industry trends. Each deal's win chance comes from a formula, including interactions. Because we wrote the formula, we know the best possible score. We call it the oracle. It is never used as a feature. Now my teammate will show the results.

---

## Speaker 2

**Slide 8 — Result A: no model beats chance.**
Here is the main result on the real data. All twelve learned models have a test ROC-AUC between 0.480 and 0.542. A value of 0.5 means random ranking. The selected model has 0.5168. One model, CatBoost on raw features, looks a little better on test. But we do not change our choice after looking at the test data. That would be cheating. The Dummy gets 60 percent accuracy just by always saying "Won". So accuracy alone is misleading.

**Slide 9 — Result B: simulated data.**
On the simulated data, the same pipeline works well. TabNet reaches 0.869, and the best possible score is 0.878. So our pipeline can learn when there is a real signal.

**Slide 10 — Same pipeline, two datasets.**
Let us put the two results together. On the real data, every model is near chance. On the simulated data with a signal, the same code reaches about 0.87. So the low real score comes from the data, not from a broken pipeline. We never mix the simulated numbers into our real results.

**Slide 11 — Scoring open deals.**
We scored 1,589 open deals. But 1,088 of them have no account information. That is 68 percent. The model never saw deals like this in training. For each deal we also show the factors that move the score. These are associations, not causes.

**Slide 12 — Where traditional ML stops.**
A traditional pipeline stops here. It gives a number and some factors. Then a person must check everything by hand. And there are hidden problems. In 827 of the 1,589 open deals, at least one input is outside the range the model saw in training. The pipeline gives no warning for that. With a weak model, missing data and hidden extrapolation, a single score is not enough to decide what to do.

**Slide 13 — Agent architecture.**
So we built an agent on top of the frozen model. It works in a loop. First it sets a goal. Then it observes the deal. Then it plans: every tool gets a score, need minus cost. It calls the best tool, and it evaluates the result. When no tool is worth calling, it decides. It can only read data, compute, and write local files. It can never email a customer or change the CRM. Every step goes into a trace.

**Slide 14 — What makes it different.**
This table compares the three systems on the same deals. The traditional pipeline gives a score. The rule-based assistant adds fixed rules, in the same order every time. Our agent chooses its own tools, based on whether the evidence could change the decision. It refuses to give a confident label when the evidence is weak. It blocks unsafe actions. And it records everything, so the decision can be replayed. Now my teammate will show it running.

---

## Speaker 3 (demo operator)

**Slide 15 — The agent on one real deal.**
Here is one real deal, 01XZ9CRY. The agent looked up the record and the account. It checked the data quality and found inputs outside the training range. It got the score, which is below the threshold. It checked the history and the explanation, and it asked the other models if they agree. The evidence was strong enough, so it created a high-priority review task. It took twelve steps, and it stopped by itself.

**Slide 16 — Live demo.**
Now I will run it live. Everything is already trained, so this takes a few seconds.
*(Run the demo. Follow the "LIVE DEMO 3–5 PHÚT" table in the demo guide. Say the sentences from that table.)*

**Slide 17 — Baseline vs agent.**
We compared the three systems on the same 1,589 real open deals and the same frozen model. The traditional pipeline gave a directional label to every out-of-support or borderline deal. That is 1,242 unsupported labels. The rule assistant gave 487. Our agent gave zero. The agent found all known data problems, while the rule assistant found 64 percent of them. Every one of the agent's evidence claims was checked again from the raw files, and all of them matched. The cost is small: about 79 milliseconds per deal, compared with 52 for the rule assistant.

**Slide 18 — Ablations and offline check.**
Then we turned off one part at a time. Without the reliability checks, 80 percent of the decisions change, and unsupported labels come back. Without planning, the decisions stay the same, but the agent calls about two more tools per deal and is almost three times slower. So planning saves cost. We also replayed the agent on 1,361 closed test deals with the outcome hidden. The deals it flagged as at risk were won 62 percent of the time, and the base rate is 60 percent. So the agent does not find losing deals better than chance. That is honest: it cannot be better than the model.

**Slide 19 — What our project adds.**
Our novelty is not a new learning algorithm. It is the agent architecture. One: it plans which evidence to collect, using value of information. Two: it gates its decisions on evidence quality and model reliability. Three: it acts on its own only inside safe, local limits. Four: every decision has a trace that we can replay. Each point is measured against two baselines and with ablations.

**Slide 20 — Limitations and future work.**
There are clear limits. The real data has little signal, so the prediction is weak. The agent's thresholds are design choices, not learned from feedback. We did not measure any increase in sales. In the future, we need engagement data like calls and emails, a test with a real sales team, and real feedback so the agent could learn.

**Slide 21 — Thank you.**
Thank you for listening. We are happy to answer your questions.

---

## Professor Q&A — short answers in simple English

**What is novel?**
We did not invent a new learning algorithm. The novelty is the agent architecture. It chooses tools by value of information. It gates its decisions on evidence quality and model reliability. It acts only inside a permission guard. And it keeps replayable traces. We measured each part against two baselines and with four ablations.

**Why is this an intelligent agent and not ordinary rules?**
Rules run the same steps every time. Our agent has a goal, a state, and a set of tools. At each step it picks the tool with the highest need minus cost. The need depends on whether the result could change its decision. So the path is different for each deal. With a missing account it uses four tools before it decides. With a full record it uses up to eight. When we removed planning, it used about two more tools per deal for the same decisions.

**Does the agent actually make decisions?**
Yes. It picks one of five actions, it creates the review task by itself, and it stops by itself. A human only decides about outside actions, like contacting a customer.

**Does it learn automatically?**
No. We have no real feedback from salespeople, so we do not claim online learning or reinforcement learning. We ran an offline replay on closed test deals with the outcome hidden. That is an evaluation, not learning.

**How is this different from ChatGPT?**
We use no language model. Every tool is a real Python function on our own data and model. The agent cannot invent a fact. Every claim it makes points to a source, and we checked all of them again from the raw files. It runs offline, for free, and it gives the same result every time.

**Why is the real-data AUC low?**
The CRM fields carry very little signal about winning or losing. Win rates are almost the same across products and sectors. There is no deal-level data like calls or budget. Also, we use an honest time-based split.

**Why is the synthetic AUC high?**
We wrote the simulated data ourselves with a strong, known signal. It only shows that our pipeline can learn when a signal exists. It says nothing about real sales.

**What does the agent improve?**
It improves how the score is used, not the score itself. It gives zero unsupported labels, compared with 1,242 for the traditional pipeline. It finds all known data problems. It leaves fewer manual steps, and every decision has a trace.

**Did you validate business uplift?**
No. We did not test with a sales team, so we do not claim more sales or revenue. In our offline replay, the deals the agent flagged were not lost more often.

**What happens when evidence is missing?**
The agent does not guess. If the account is missing, it asks to fix the data first. If a tool fails, it tries once more, then it stops and records the error. It never makes up a probability.

**Why should this be considered excellent-level work?**
We hope it meets the criteria for novelty in the agentic part, but you will judge that. We built a working intelligent agent with its own planning, safety rules and traces. We tested it on real data against two baselines and with ablations. We also report the weak prediction results honestly. All the code runs with one command on a normal laptop.
