# Natural ESL presentation script

Suggested allocation: 12 slides / 3 members / 4 slides each. About 8–10 minutes total at a comfortable speaking pace. Swap the names if the group prefers; these are speaking assignments, not claims about past contributions.

## Slide 1: CRM sales opportunities

**Hong Thai Phan**

Hello everyone. Our project is about predicting whether a CRM sales opportunity will be won or lost. We also want to show which inputs affect the model score. We have now completed the comparison of the models in our proposal. The main finding is that the available data gives us only a weak prediction signal. So today we will explain both the working system and its limits. We will start with the data, then show the evaluation, and finish with the outputs and next steps.

## Slide 2: What goes into the decision?

**Hong Thai Phan**

We use four connected tables. The opportunity table gives us the product, sales agent and engagement date. The account table adds information such as industry, revenue and company size. The other tables add product price and sales team information. After preparation, we use eighteen predictors. We do not use the final close value, close date or deal stage as prediction inputs, because they can reveal the answer. We keep the opportunity ID only to connect a prediction back to its original record.

## Slide 3: Data prepared and checked

**Hong Thai Phan**

The data contains 8,800 opportunities. We use the 6,711 closed records for supervised learning. The remaining records are still open. We score the 1,589 Engaging records, but leave out the 500 Prospecting records. We also fixed inconsistent product names and checked the table joins. An important issue is missing account information: 1,088 scored records have this warning. We make that visible because missing inputs can reduce how much we can trust a model score.

## Slide 4: Respect what was known at the time

**Hong Thai Phan**

We split by engagement date, but date order alone is not enough. Imagine a deal starts in June but closes in October. Its outcome was not known in July, so it should not train a model used at that time. We exclude records like this from the earlier training or validation period. Every model uses the same split. We also keep deals from the same date together. This is still an exploratory historical test, because we have already studied this dataset. Now my teammate will explain the models and results.

## Slide 5: Compare all promised models

**Nguyen Khanh An Tran**

We trained all four models from our proposal, plus a simple control model. The control always gives the training win rate. Logistic Regression is our linear baseline, and Random Forest can learn more complex patterns. We also trained an MLP and TabNet on CPU. All preprocessing is fitted on the training set only. Validation data controls early stopping, model selection and the decision threshold. We did not choose the model using the test results. The saved settings let us repeat the comparison.

## Slide 6: Results: weak discrimination

**Nguyen Khanh An Tran**

Here are the actual test results. Logistic Regression was selected on validation, and its test AUC is about 0.52. An AUC of point five is the no-ranking baseline, so this result is only slightly above that level. The more advanced models did not give a convincing improvement in this run. Also, a model that predicts mostly Won can have a fairly high accuracy or F1 because Won is the larger class. That is why we show the control and several metrics together.

## Slide 7: A perfect score can be the wrong answer

**Nguyen Khanh An Tran**

We also ran a deliberately incorrect experiment to show the effect of data leakage. In these files, lost deals have zero close value. If we use that field, even a very small decision tree can almost directly recover the outcome. The score looks excellent, but the information is only known after the deal closes. So this would not help us predict a new opportunity. We keep this experiment only as an audit example. It is never used in the real prediction output.

## Slide 8: Show what influences the score

**Nguyen Khanh An Tran**

For each scored opportunity, we show which inputs increase or decrease the model score compared with a training reference. For example, we replace one numeric value with its training median and measure the probability change. These changes help explain the model, but they are not additive SHAP values. Separately, we computed TreeSHAP for the raw Random Forest and checked that its contributions reconstruct the prediction. We clearly label the two methods. Neither method proves cause and effect. My teammate will now show the output and practical limits.

## Slide 9: Two outputs, with an honest context

**Hoang Thien Bao Bui**

The first output is the estimated win probability and its complementary loss probability. This real open record has about 88% predicted win probability. The second output explains the main input factors affecting that score. The CSV also includes a predicted class, a simple priority label and missing-data warnings. We should be careful with this example: the actual outcome is unknown, and this is a snapshot demonstration. We are not claiming that this score was available on the original engagement date, or that the priority label has proven business value.

## Slide 10: Calibration helps scale, not ranking

**Hoang Thien Bao Bui**

We used validation data to adjust the probability scale. On the test set, we compare the Brier score before and after calibration. A lower score is better, but calibration does not improve the ordering of deals. The base model stays frozen so the calibration still matches it. Our priority cutoffs are only examples and need business validation. We also have limits from using only observed closed outcomes, changes over time and missing account information. So this version should be treated as a research demonstration, not a production sales tool.

## Slide 11: What is complete?

**Hoang Thien Bao Bui**

The main items from our proposal now have working code and generated outputs. We have data checks, exploratory analysis, all four promised models, the comparison metrics and confusion matrices. We also completed feature importance and the optional SHAP analysis. The workflow creates the open-deal CSV, figures, report and slides from the same run. This helps keep our presentation consistent with the code. The remaining checks involve the group account and final presentation review. We still need to confirm the notebook in our own Colab session and check the slides after import.

## Slide 12: Demo, review and next steps

**Hoang Thien Bao Bui**

For the demo, we can upload the project ZIP, run the notebook and inspect the generated results. We should check the run status, compare the models and open one scored opportunity with its explanation. Before submission, each member should review the report, rehearse their part and confirm the instructor schedule. The code is prepared for review in a pull request to our group repository. We should review the changes before merging. For future research, more useful pre-close activity data and a prospective test would matter more than simply adding another complex model. Thank you. We are happy to take questions.

## Short Q&A practice

**What is the novelty?** We do not claim a new algorithm. Our contribution is an explainable CRM workflow where every safeguard has evidence. The leaky control shows how an outcome field fakes a perfect score. We compared our time-based split with a random split. We checked that the explanations really describe the model. A known-truth simulation shows the pipeline learns when signal exists.

**Does your time-based split really matter?** On this dataset, the effect is small. A random split gives an AUC up to 0.04 higher, but every protocol is still close to chance. We measured it instead of assuming it. The much bigger danger is the close value field.

**How do you know the explanations are correct?** We tested them against the model. If we reset a deal's top three factors, the score moves about 3 times more than if we reset three random factors. They also agree with SHAP on the Random Forest. This shows they describe the model, not that they are real causes.

**Why is the accuracy low?** The available pre-close features contain little useful signal in this test. We prefer to report that honestly instead of using information that is only known after the outcome.

**Why did you use TabNet?** We promised to compare traditional models with an advanced tabular model. TabNet gives us that comparison, but it did not show a useful improvement here.

**Can the sales team use the priority score now?** It is a demonstration at this stage. We need better data coverage and a prospective test before using it to allocate sales effort.

**What is the difference between SHAP and your local explanation?** Our main output changes one input to a training reference and measures the score change. Those changes do not add up to the prediction. TreeSHAP is a separate additive explanation for the raw Random Forest.

**What have you actually run?** We ran all five models on CPU in the development environment. The notebook uses the same code. The group still needs to confirm the run in its own Colab account.

**Is this an intelligent agent?** This version is a supervised decision-support pipeline. The proposal did not require an autonomous agent, and we do not claim one.

**Does a 70 percent score mean this particular deal will win?** No. It is a model estimate, not a guarantee. Calibration, missing inputs and changes in the data all affect how much we can trust it.
