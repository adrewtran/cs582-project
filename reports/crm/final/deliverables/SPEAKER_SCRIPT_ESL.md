# CRM presentation script

Suggested speaking allocation only. This does not claim each member performed these development tasks. Target 11 minutes (allow 10–12 minutes) including a short live command. Speaking order: Hong Thai Phan (slides 1–4, 0:00–3:40), Nguyen Khanh An Tran (slides 5–8, 3:40–7:20), Hoang Thien Bao Bui (slides 9–12, 7:20–11:00). This is presentation allocation, not a claim about development authorship. Aim for 100–115 words per minute. Rehearse once with a timer. The timings include the live demonstration. Questions can follow after the 11 minutes.

## Slide 1 CRM sales decision support

Hong Thai Phan | 0:00 to 0:45

Hello everyone. Our project studies sales opportunities in a CRM system. We want to estimate whether an opportunity will be won or lost, and help the user understand the score. Our system now includes a sales assistant that suggests review actions with evidence. Today we will show the problem, our data, the experiment, and a real model demo. We will also explain the limits of the measured results. These matter, because we want our conclusions to match what the data actually shows.

## Slide 2 The prediction task and its inputs

Hong Thai Phan | 0:45 to 1:40

The source is a fictitious business that sells computer hardware. We connect four tables: sales opportunities, accounts, products, and sales teams. There are 8,800 opportunities in the supplied files. We learn from observed Won and Lost outcomes. An open deal is still unknown, so we never label it as Lost. The raw model uses eighteen pre-close attributes. The final close value and stage would reveal the answer. We exclude those fields from prediction, while keeping dates and IDs for auditing.

## Slide 3 A split that respects label availability

Hong Thai Phan | 1:40 to 2:40

We split the records by engagement date. However, starting earlier does not mean the outcome was already known. For example, a deal might start in June and close in October. Its outcome cannot train a model used in July. Our split removes these late labels from the earlier periods. We also keep records from the same date together. All models use the same row assignments. This is an exploratory test, because we already inspected this dataset during the project. It is not a new external test.

## Slide 4 Historical features without future outcomes

Hong Thai Phan | 2:40 to 3:40

Our first extension is historical information. For each opportunity, we look backward from its engagement date. We only use other training deals that had already closed. A closure on the same day is excluded, because we do not know the time within that day. We calculate prior counts, smoothed win rates, average values, and cycle lengths. When history is missing, we use a documented prior and show a cold-start flag. Tests confirm that changing future outcomes does not change these features. An will now explain how we compare the models.

## Slide 5 A controlled model comparison

Nguyen Khanh An Tran | 3:40 to 4:35

We compare six real models on CPU. The original models are Logistic Regression, Random Forest, MLP, and TabNet. We retain the dummy control and add CatBoost, which handles categorical data directly. Experiment A uses the original inputs. Experiment B adds the historical features. We fit preprocessing using training data only. Validation controls selection and early stopping. We save a selection record before evaluating either test set. This means a model with a better test score cannot silently become our final choice after we see the result.

## Slide 6 The measured effect of historical features

Nguyen Khanh An Tran | 4:35 to 5:40

Here are the measured test AUC values. For Logistic Regression, history has lower test AUC than raw inputs (B minus A -0.0070). This single exploratory comparison does not prove a general predictive benefit. The highest observed test AUC is 0.542, from raw / CatBoost. That is descriptive information, not our selection rule. We keep the validation winner. We do not switch models after seeing test results. The paper also includes accuracy, precision, recall, F1, calibration scores, and confusion counts for all twelve legitimate runs. This helps us judge the models from several perspectives.

## Slide 7 The leakage audit checks an invalid shortcut

Nguyen Khanh An Tran | 5:40 to 6:30

We also run an intentionally invalid control. Its test AUC is 1.0000 using the final close value. However, the salesperson would not know that value when making the prediction. We keep this control outside the real model comparison and scoring. Historical average values are different: they come from other deals that already closed before the query date. This audit shows why checking data availability is more important than presenting an impressive number. Even if the diagnostic score changes on new data, this field remains invalid for pre-close prediction.

## Slide 8 The sales assistant has inspectable rules

Nguyen Khanh An Tran | 6:30 to 7:20

The assistant has two separate parts. The model estimates a probability and shows the inputs that affect its score. The rule layer then suggests review actions using that score and the historical context. For example, it asks the user to complete a missing account, or review product fit when enough prior product outcomes show a weaker pattern. Each action includes the rule name, observed evidence, and reason. These are transparent decision-support rules. We do not claim they cause more sales. Bao will now demonstrate a real opportunity.

## Slide 9 A real opportunity through the saved model

Hoang Thien Bao Bui | 7:20 to 8:50

For this demo, we use opportunity 01XZ9CRY, for Initech. I will run the demo command now. It loads the saved model, rebuilds the same historical features, and calculates the score again. The estimated win probability is about 53%. The actual outcome is still unknown. Below the score, we can see the strongest reference sensitivities and the assistant's review actions. The JSON version includes the supporting counts and timestamps. We choose this example by date, not because its score looks especially good.

Demo cue: run the prepared src.demo command. Show the probability, one reason, and the evidence beneath one action. Allow about 20 seconds for the screen walkthrough.

## Slide 10 Reasons and evidence quality

Hoang Thien Bao Bui | 8:50 to 9:40

An explanation can make a weak model easier to inspect, but it cannot make that model accurate. Our main explanations change one input to a training reference and measure the score difference. These effects do not add up like SHAP values, and correlated features can make such changes unrealistic. We also keep a separate Random Forest TreeSHAP audit. Missing account information is another major issue in the open records. We show this warning explicitly. The assistant still requires human review and does not turn missing data into confident business advice.

## Slide 11 The contribution and its limits

Hoang Thien Bao Bui | 9:40 to 10:25

Our novelty is an applied system contribution. We combine a leakage audit, a tested historical feature engine, and a deterministic sales assistant with traceable evidence. We are not presenting a newly invented learning algorithm. The paired comparison shows the measured effect of adding history for this dataset. It does not establish a general improvement. The training archive is frozen, the account tables are static, and the data contains only observed closed outcomes for evaluation. Stronger future evidence would require richer activity data, new periods, and a prospective study of actual decisions.

## Slide 12 Reproducible delivery and next steps

Hoang Thien Bao Bui | 10:25 to 11:00

To finish, one command runs the full experiment and creates our results. The notebook uses the same code as this demo. We check the time rules and confirm that saved models reproduce the exported scores. Our next step is to test richer sales activity data and evaluate new opportunities. We will also rehearse in Colab and check the slide layout before presenting. Thank you. We are happy to answer your questions.

## Natural ESL answers for questions
Why can a win probability above 50 percent still have a Lost label? We choose the classification threshold on validation data. It is about 61 percent for this saved run, rather than a fixed 50 percent. The probability, class label and loss-risk band answer different questions.


What is new here? Our contribution combines a tested time-aware history engine, a leakage audit, and a sales assistant that explains each suggested review action. We use existing learning algorithms. We do not claim a new algorithm or proven sales improvement.

Did history improve accuracy? For Logistic Regression, history has lower test AUC than raw inputs (B minus A -0.0070). This single exploratory comparison does not prove a general predictive benefit. We report that result directly.

Why not select the highest test score? The highest observed test AUC is 0.5424 from raw / CatBoost, but the selection rule uses validation. Changing the rule after seeing test results would make our evaluation less trustworthy.

Is this an agent or a chatbot? It is a deterministic decision-support agent. It observes a prediction and evidence, applies explicit rules, and recommends review actions. It has no LLM, autonomous planning, or ability to contact customers.

Why are past close values allowed? They belong to other training deals that closed before the query date. The current opportunity's final value is never used by the legitimate predictor. The audit records the latest eligible past closure.

What does the score mean? It is an estimated probability from a model trained on observed closed deals. It is not a guarantee for this opportunity, and it may not transfer to open deals with missing information.

Does the explanation prove why a sale is lost? No. It shows model sensitivity to input values. It does not prove a real causal effect or that changing the input will improve the outcome.

Why are the results weak? This snapshot has limited pre-close information. Historical sparsity, changing feature distributions and time drift may contribute. We have not isolated those causes experimentally.

How do you validate the agent? Tests check whether each rule fires under its stated evidence, whether actions are deterministic, and whether the live model matches the exported score. A user study and prospective intervention study remain future work.

Can we call this excellent? The implementation adds testable novelty and a reproducible demonstration. The professor decides the grade. Weak predictive performance and unvalidated business value remain limitations, so we should not promise an excellent grade.
