# Credit Pre-Screening Engine --- Loan Intake Optimisation 🔍

A machine-learning pre-screening prototype for unsecured personal-loan
applications. It is designed to help a bank prioritise analyst
attention, reduce avoidable manual review, and support consistent intake
decisions.

> **Important:** The target in this project is the historical analyst
> decision (`Approved` / `Declined`). It is not a repayment,
> delinquency, or default outcome. This is a decision-support prototype,
> not an autonomous lending system.

## Table of Contents

-   [1. Business Problem](#1-business-problem)
-   [2. Objectives](#2-objectives)
-   [3. Dataset](#3-dataset)
-   [4. End-to-End Workflow](#4-end-to-end-workflow)
-   [5. Data Cleaning](#5-data-cleaning)
-   [6. Exploratory Data Analysis](#6-exploratory-data-analysis)
-   [7. Feature Engineering](#7-feature-engineering)
-   [8. Machine Learning Model](#8-machine-learning-model)
-   [9. Model Evaluation](#9-model-evaluation)
-   [10. Threshold Selection and Business
    Impact](#10-threshold-selection-and-business-impact)
-   [11. Streamlit Application](#11-streamlit-application)
-   [12. Run Locally](#12-run-locally)
-   [13. Business Recommendations](#13-business-recommendations)
-   [14. Key Takeaways](#14-key-takeaways)
-   [15. Limitations and Future Work](#15-limitations-and-future-work)
-   [16. Project Structure](#16-project-structure)

## 1. Business Problem

A mid-sized private bank processes approximately **3,000 unsecured
personal-loan applications per month**. Manual review takes around **25
minutes per application**. A substantial number of applications are
declined based on relatively clear indicators, such as low credit scores
or high EMI burden.

Reviewing every application manually consumes analyst capacity and can
delay applications that require deeper judgement. The business needs a
pre-screening approach that identifies high-confidence cases for
prioritised handling while keeping uncertain and borderline cases with
credit analysts.

## 2. Objectives

-   Understand the dataset and assess data quality.
-   Identify applicant characteristics associated with historical
    decisions.
-   Build a binary classification model.
-   Evaluate class-level performance and error types.
-   Select a conservative decision threshold using a business
    constraint.
-   Estimate potential analyst capacity released.
-   Demonstrate predictions through a Streamlit interface.

## 3. Dataset

The project uses the simulated **MUC02 Credit Pre-Screening Dataset**.

  Attribute   Description
  ----------- ----------------------------------
  Records     12,000 applications
  Columns     19
  Period      April--September 2026
  Target      `decision`
  Classes     `Approved`, `Declined`
  ML task     Supervised binary classification

Relevant fields include `credit_score`, `monthly_income`,
`existing_emi`, `proposed_emi`, `loan_amount`, `tenure_months`,
`delinquencies_24m`, `num_active_loans`, `years_in_current_job`,
`employment_type`, `loan_purpose`, `city_tier`, and `decision`.

The target is the historical analyst decision. It must not be
interpreted as evidence that an applicant repaid or defaulted on a loan.

## 4. End-to-End Workflow

``` text
Business Problem Definition
          |
          v
Dataset Loading and Understanding
          |
          v
Data Quality Checks and Cleaning
          |
          v
Exploratory Data Analysis (EDA)
          |
          v
Feature Engineering
          |
          v
Stratified Train-Test Split (80:20)
          |
          v
Numerical and Categorical Preprocessing
          |
          v
Logistic Regression Training
          |
          v
Model Evaluation
          |
          v
Decision Threshold Analysis
          |
          v
Business Recommendations
          |
          v
Streamlit Pre-Screening Application
```

## 5. Data Cleaning

The project applies the following preparation rules:

1.  **Missing monthly income:** Impute using the median for the
    applicant's employment type, with an overall median fallback.
2.  **Missing credit score:** Retain these applicants and represent
    missing score status using an `is_ntc` indicator rather than
    dropping the rows.
3.  **Unusually high monthly income:** A small number of values above
    ₹600,000 were treated as possible annual-income entries and divided
    by 12, based on the project analysis.
4.  **Data preparation:** Check data types and prepare numerical and
    categorical columns for modelling.
5.  **Target:** Use `decision` as the supervised learning label.

These are project-specific assumptions and should be checked against
real bank data-entry rules before deployment.

## 6. Exploratory Data Analysis

EDA was used to understand how applicant and loan characteristics relate
to historical approval decisions.

### Main observations

-   **Credit score:** Historical approval rates were very low below 600
    and remained low in the 600--649 band. Approval increased above the
    650 boundary.
-   **No recorded credit score (NTC):** This segment showed a different
    approval pattern and should not automatically be treated as
    high-risk.
-   **Employment type:** Salaried applicants had a higher historical
    approval rate than Gig / Contract applicants in this dataset.
-   **FOIR:** Approval rates declined as the EMI burden increased, with
    a steeper decline beyond approximately 50%.

### FOIR

Fixed Obligation to Income Ratio estimates the share of monthly income
committed to EMI payments:

\[ FOIR = `\frac{Existing\ EMI + Proposed\ EMI}{Monthly\ Income}`{=tex}
\]

A higher FOIR indicates a greater monthly repayment burden relative to
income.

## 7. Feature Engineering

Three derived features were created:

  Feature            Definition
  ------------------ --------------------------------------------------
  `foir`             `(existing_emi + proposed_emi) / monthly_income`
  `loan_to_income`   `loan_amount / (monthly_income * 12)`
  `is_ntc`           `1` when credit score is missing; otherwise `0`

### Model inputs

Numerical features: - `credit_score` - `is_ntc` - `foir` -
`loan_to_income` - `delinquencies_24m` - `num_active_loans` -
`years_in_current_job` - `age` - `tenure_months`

Categorical features: - `city_tier` - `employment_type` - `loan_purpose`

The model excludes `application_id`, `application_date`, `city`, and
`gender` as predictors. The target `decision` is also excluded from the
input features.

## 8. Machine Learning Model

### 🎯Selected model: Logistic Regression

This is a **supervised binary classification** problem because the
target has two classes: `Approved` and `Declined`. We use **Logistic
Regression** as the baseline classifier.

### 🎯Why Logistic Regression?

1.  **Fits binary classification:** It estimates the probability of an
    application belonging to a class.
2.  **Interpretable:** Coefficients indicate the direction of
    association between input features and the predicted class. This
    helps explain model behaviour to credit-policy stakeholders.
3.  **Supports threshold tuning:** Its probability output allows the
    business to test different decision thresholds instead of relying
    only on the default 0.50 cutoff.
4.  **Good baseline:** It is comparatively simple to train, evaluate,
    explain, and maintain.
5.  **Handles mixed inputs with preprocessing:** Numerical and
    categorical variables can be combined using scaling and one-hot
    encoding.

We selected Logistic Regression for explainability and threshold-based
decision support, not because it is guaranteed to outperform all other
algorithms. More complex models should be compared against this baseline
using the same validation approach and business constraints.

### Training pipeline

-   Numerical columns: median imputation followed by `StandardScaler`.
-   Categorical columns: most-frequent imputation followed by
    `OneHotEncoder`.
-   Train-test split: 80:20 with stratification.
-   Random state: `42`.
-   Classifier: `LogisticRegression(max_iter=2000)`.

The transformations and estimator are combined in a scikit-learn
pipeline so that prediction-time preprocessing is consistent with
training.

## 9. Model Evaluation

The held-out test set contains 2,400 applications.

  Metric                         Reported result
  ------------------------ ---------------------
  Accuracy                     Approximately 79%
  Precision for Declined     Approximately 79.1%
  Recall for Declined        Approximately 75.5%
  ROC-AUC                    Approximately 0.867

The positive class in this evaluation is `Declined`.

### Confusion matrix

  Actual / Predicted     Approved   Declined
  -------------------- ---------- ----------
  Actual Approved           1,046        225
  Actual Declined             277        852

-   **1,046 true negatives:** Historically Approved applications
    correctly predicted as Approved.
-   **225 false positives:** Historically Approved applications
    predicted as Declined. These are wrong-decline cases and are
    particularly important in this use case.
-   **277 false negatives:** Historically Declined applications
    predicted as Approved. They may proceed to further review if used in
    a pre-screening workflow.
-   **852 true positives:** Historically Declined applications correctly
    predicted as Declined.

Accuracy alone is insufficient. Precision, recall, confusion-matrix
errors, and threshold-specific wrong-decline rates are needed to
understand operational risk.

## 10. Threshold Selection and Business Impact

Logistic Regression outputs a probability. A threshold determines when
an application is assigned to the `Declined` class. The default 0.50
threshold is not necessarily suitable for this business workflow.

The project tested higher thresholds and selected **0.90** as a
conservative threshold under the stated constraint:

-   An application enters the prototype auto-decline lane only when
    predicted `Declined` probability is at least 0.90.
-   Applications below the threshold remain in the analyst-review
    workflow.
-   The proportion of historically Approved applicants in the
    auto-decline lane must remain within the project's **5%
    wrong-decline limit**.

### Results at the selected threshold

  -----------------------------------------------------------------------
  Measure                                                          Result
  ------------------------------ ----------------------------------------
  Wrong-decline proportion                             Approximately 4.7%
  within auto-decline lane       

  Applications entering                               Approximately 15.2%
  auto-decline lane              

  Estimated auto-declines per                                         456
  month                          

  Manual review time per                                       25 minutes
  application                    

  Potential analyst time                                  190 hours/month
  released                       
  -----------------------------------------------------------------------

Estimated monthly capacity:

\[ 456 `\times 25`{=tex} `\div 60`{=tex} = 190 hours \]

This is a potential capacity estimate, not a guaranteed saving. Realised
savings depend on operating procedures, exception handling, mandatory
reviews, and live model performance. The 0.90 threshold must be
independently validated before operational use.

## 11. Streamlit Application

A Streamlit interface demonstrates the end-user flow:

``` text
Applicant details entered
          |
          v
Input validation
          |
          v
Feature engineering (FOIR, loan-to-income, NTC)
          |
          v
Load saved scikit-learn pipeline
          |
          v
Generate class prediction and probabilities
          |
          v
Display pre-screening result
          |
          v
Analyst / policy review where required
```

The application accepts applicant, employment, income, credit-history,
and loan details. It calculates derived features, loads the trained
model pipeline, and displays the prediction and probability information.

This is a demonstration interface. It is not connected to a bank loan
origination system and does not make a legally binding lending decision.

## 12. Run Locally

### Prerequisites

-   Python 3.9+
-   Jupyter Notebook
-   Streamlit
-   pandas
-   numpy
-   scikit-learn
-   joblib

Install dependencies:

``` bash
pip install pandas numpy scikit-learn joblib streamlit jupyter
```

### Step 1 --- Prepare the project files

Ensure the dataset is available to the notebook:

``` text
MUC02_Credit_PreScreening_Dataset.csv
```

### Step 2 --- Train and save the model

Open and run every cell in:

``` text
Loan_Approve_Decline_Deployment_Model.ipynb
```

The notebook trains the preprocessing and Logistic Regression pipeline
and creates:

``` text
loan_approve_decline_pipeline.joblib
```

### Step 3 --- Start Streamlit

Keep `app.py` and `loan_approve_decline_pipeline.joblib` in the same
directory, then run:

``` bash
streamlit run app.py
```

Open the local URL printed in the terminal.

**Consistency check:** The feature engineering and category values in
`app.py` must match the notebook. In particular, the definition of
`is_ntc` and the employment-type and loan-purpose categories must be
identical during training and prediction.

## 13. Business Recommendations

1.  **Use the model for pre-screening, not autonomous lending.** Keep
    credit policy and accountable human decision-making in control.
2.  **Route borderline cases to analysts.** The conservative threshold
    should create a limited high-confidence lane; other applications
    should follow the existing review process.
3.  **Treat NTC applicants carefully.** Model performance was materially
    weaker for applicants without recorded credit scores than for scored
    applicants. Route this segment to human assessment rather than
    applying the same automated decline rule.
4.  **Review affordability guardrails.** The EDA indicates a marked
    approval decline as FOIR rises, particularly beyond roughly 50%. A
    50--55% range may be worth reviewing as a policy guardrail, but it
    is not a universal lending rule.
5.  **Monitor error rates over time.** Track wrong-decline rate, false
    negatives, precision, recall, and threshold performance---not just
    accuracy.
6.  **Monitor fairness.** Excluding gender from model inputs does not
    eliminate proxy effects. Income and employment type may reflect
    socioeconomic differences; evaluate outcomes across relevant groups.
7.  **Obtain repayment outcomes.** Historical analyst decisions cannot
    establish whether a loan will be repaid. A genuine credit-risk model
    requires repayment, delinquency, or default labels.
8.  **Revalidate regularly.** Applicant populations, credit policy,
    economic conditions, and data collection practices can change.

## 14. Key Takeaways

-   The use case is a **binary classification** problem.
-   **Logistic Regression** was selected as an explainable baseline with
    probability outputs that support threshold tuning.
-   Data preparation and feature engineering---especially FOIR and the
    NTC indicator---are important to the workflow.
-   Model quality must be judged by the consequences of false declines
    and false approvals, not accuracy alone.
-   A 0.90 threshold produced an estimated 190 analyst hours of monthly
    capacity, subject to validation.
-   NTC and borderline applicants require careful human review.
-   The model predicts historical analyst decisions; it does **not**
    predict repayment or default.

## 15. Limitations and Future Work

### Limitations

-   The dataset is simulated and may not represent real applicant
    behaviour.
-   The target is a historical analyst decision, not an observed
    repayment outcome.
-   Model performance differs across segments, particularly for
    applicants without recorded credit scores.
-   The threshold and estimated time savings require independent
    validation.
-   Removing gender from the features does not, by itself, guarantee
    fairness.

### Future improvements

-   Validate using newer, out-of-time application data.
-   Obtain repayment and default outcomes for a genuine credit-risk
    modelling objective.
-   Compare Logistic Regression with tree-based models using consistent
    validation and business-cost criteria.
-   Add probability calibration and threshold monitoring.
-   Complete documented fairness, explainability, privacy, and
    compliance reviews.
-   Introduce human-review feedback and ongoing model monitoring before
    production rollout.

## 16. Project Structure

Suggested GitHub repository layout:

``` text
credit-pre-screening-engine/
│
├── MUC02_Credit_PreScreening_Dataset.csv
├── Loan_Approve_Decline_Deployment_Model.ipynb
├── app.py
├── loan_approve_decline_pipeline.joblib   # generated by the notebook
├── requirements.txt
└── README.md
```

Example `requirements.txt`:

``` text
pandas
numpy
scikit-learn
joblib
streamlit
```

------------------------------------------------------------------------

## Conclusion

The Credit Pre-Screening Engine demonstrates how an interpretable
machine-learning model can support loan-intake operations by identifying
a conservative group of applications for prioritised handling. Logistic
Regression was selected because it provides probability estimates and
interpretable feature relationships, making it a practical baseline for
a policy-sensitive workflow. A high threshold was used to respect the
project's wrong-decline constraint while estimating a possible reduction
in manual workload.

The recommended next step is a controlled pilot with human oversight,
independent validation, fairness monitoring, and actual repayment data.
The goal is to improve analyst productivity and applicant experience
without allowing a prototype model to replace responsible credit
judgement.
