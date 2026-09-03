# **Machine Learning CS 582: Project Proposal**

## **Group 2**

## **Project title**

Predicting CRM Sales Opportunities Using Machine Learning

## **Data set**

We will use the CRM Sales Opportunities dataset from Maven Analytics
(https://mavenanalytics.io/data-playground/crm-sales-opportunities). The dataset contains information about sales
opportunities, customer accounts, products, and sales teams. We will join the related tables and use the available
historical information to build features for predicting the final sales outcome (Won or Lost).

## **Project idea**

The objective of this project is to build a binary classification pipeline that predicts whether a CRM sales opportunity
will be Won or Lost. A useful prediction can help a sales team prioritize opportunities, allocate resources more
effectively, and focus attention on deals with a higher probability of success. We will prepare the CRM data by joining
the related tables, cleaning missing or inconsistent values, selecting useful features, and splitting the data into
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

We will write Python programs to load and join the CRM data, clean and preprocess the data, create and select features,
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

- [5] Maven Analytics, “CRM Sales Opportunities,” Data Playground. [Online].
  Available: https://www.mavenanalytics.io/data-playground

2 

