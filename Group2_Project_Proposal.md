# **Machine Learning CS 582: Project Proposal**

## **Group 2**

## **Project title**

Predicting Sales Lead Conversion Using Machine Learning

## **Data set**

Our primary dataset is the Lead Scoring dataset from X Education, an online education company (available on Kaggle,
https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset). It contains 9,240 sales leads with 37
attributes describing how each lead was acquired (lead origin and source), the lead's activity on the website (visits,
time spent, pages per visit, last activity) and profile information from the sign-up form (occupation, specialization,
city), together with a binary label indicating whether the lead converted into a paying customer (38.5% converted).
As a secondary dataset for checking whether our findings generalize, we will use the UCI Bank Marketing dataset
(41,188 telemarketing contacts, 11.3% positive), which is larger and strongly imbalanced. Both datasets contain
columns that are only known after the outcome (for example the sales team's post-contact tags in the lead data, and
the call duration in the bank data); identifying and excluding such leakage columns is part of our data preparation.
We initially considered the Maven Analytics CRM Sales Opportunities dataset, but found that its attributes carry no
predictive signal for the deal outcome (ROC-AUC ≈ 0.5 without leakage columns); we will report this as a short
leakage-audit case study.

## **Project idea**

The objective of this project is to build a binary classification pipeline that predicts whether a sales lead
will convert into a customer. A useful prediction can help a sales team prioritize opportunities, allocate resources more
effectively, and focus attention on deals with a higher probability of success. We will prepare the lead data by
removing leakage columns, cleaning missing or inconsistent values, selecting useful features, and splitting the data into
training and testing sets.

We will first establish baseline results with traditional machine learning methods such as Logistic Regression and
Random Forest. We will then evaluate more advanced models, including a Multi-Layer Perceptron (MLP) and TabNet for
tabular data. The models will be compared using Accuracy, Precision, Recall, F1-score, Confusion Matrix, and ROCAUC. In
addition to comparing predictive performance, we plan to examine the factors that contribute to the predictions so that
the results are more useful and understandable for sales decision-making.

## **Novelty**

The novelty of our project is to combine sales opportunity prediction with model explainability instead of only
reporting whether a deal is likely to be won or lost. After building and comparing the prediction models, we will
analyze which features have the greatest influence on the results. We plan to use model feature importance and, if time
permits, SHAP-based explanations to show why an opportunity receives a high or low predicted win probability. We will
also compare traditional machine learning models with TabNet to study whether an advanced tabular model can provide
better predictive performance while still offering useful information about important features. This can make the
project more practical because a sales team can understand not only the prediction, but also the main factors behind it.

## **Software you will need to write**

We will write Python programs to load the datasets, clean and preprocess the data, create and select features,
split the data into training and testing sets, train the machine learning models, evaluate and compare the results, and
generate charts. We

1

plan to use Pandas, NumPy, Matplotlib, Scikit-learn, and additional Python packages needed for TabNet and model
explainability.

## **Papers to read**

- J. Yan et al., “Sales Pipeline Win Propensity Prediction: A Regression Approach,” IFIP/IEEE IM, 2015.

- A. Rezazadeh, “A Generalized Flow for B2B Sales Predictive Modeling: An Azure Machine-Learning Approach,” Forecasting,
  2020.

- S. O. Arik and T. Pfister, “TabNet: Attentive Interpretable Tabular Learning,” AAAI Conference on Artificial
  Intelligence (AAAI-21), 2021.

## **Teammate**

Yes. Group 2 team members are:

- Hong Thai Phan

- Nguyen Khanh An Tran

- Hoang Thien Bao Bui

## **What will you complete by the Tuesday of 4th week?**

By the Tuesday of the 4th week, we plan to complete the initial data cleaning and table joins, perform basic exploratory
data analysis, and select the first set of features. We will also train at least two baseline models, Logistic
Regression and Random Forest, and report initial experimental results using Accuracy, Precision, Recall, F1-score,
Confusion Matrix, and ROC-AUC. These early results will give us a baseline that we can later compare with MLP and
TabNet.

## **References**

- [1] S. Marsland, Machine Learning: An Algorithmic Perspective, 2nd ed. Boca Raton, FL, USA: CRC Press, 2015.

- [2] J. Yan et al., “Sales Pipeline Win Propensity Prediction: A Regression Approach,” in Proc. IFIP/IEEE Int. Symp.
  Integrated Network Management (IM), Ottawa, Canada, 2015, pp. 854–857.

- [3] A. Rezazadeh, “A Generalized Flow for B2B Sales Predictive Modeling: An Azure Machine-Learning Approach,”
  Forecasting, 2020.

- [4] S. O. Arik and T. Pfister, “TabNet: Attentive Interpretable Tabular Learning,” in Proc. AAAI Conf. Artificial
  Intelligence (AAAI-21), vol. 35, no. 8, 2021, pp. 6679–6687.

- [5] A. Chatterjee, “Lead Scoring Dataset,” Kaggle. [Online].
  Available: https://www.kaggle.com/datasets/amritachatterjee09/lead-scoring-dataset

- [6] S. Moro, P. Cortez, and P. Rita, “A Data-Driven Approach to Predict the Success of Bank Telemarketing,”
  Decision Support Systems, vol. 62, pp. 22–31, 2014. Dataset: UCI Machine Learning Repository, “Bank Marketing.”

- [7] Maven Analytics, “CRM Sales Opportunities,” Data Playground. [Online].
  Available: https://www.mavenanalytics.io/data-playground

2 

