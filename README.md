# CodeAlpha Credit Scoring Model

A machine learning project built for the CodeAlpha Machine Learning internship (Task 1: Credit Scoring Model).

**Goal:** predict whether a borrower will be 90 days or more late on a payment within two years, using their past financial history.

## Results at a glance

| Model | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.216 | 0.736 | 0.334 | 0.858 |
| **Random Forest (final model)** | **0.220** | **0.753** | **0.341** | **0.862** |

All scores are for the defaulter class (1) on a held-out test set. The Random Forest is slightly ahead on all four metrics, but the gap is small, so both models perform similarly.

## Dataset

- **Name:** Give Me Some Credit
- **Source:** Kaggle competition, https://www.kaggle.com/c/GiveMeSomeCredit
- **File used:** `cs-training.csv` only. It is the only file that includes the answers (the target). The test file and sample entry file were not used.
- **Size:** about 150,000 borrowers, 10 input columns plus the target
- **Target:** `SeriousDlqin2yrs`. 1 means the person was 90 days or more late on payments, 0 means they were not.
- **Class balance:** only about 7% of borrowers are 1, so the data is very uneven.
- **Downloaded:** [date]
- The data belongs to the competition and is used here for learning. It is not included in this repo, so please download it from the link above and place it in the `data/` folder.

## Approach

1. **Data audit.** Checked every column for missing values, duplicates, impossible values and odd codes, and compared the columns with the data dictionary.
2. **Basic cleaning (before the split).** These fixes use no numbers learned from the data.
   - Removed the duplicate rows that had decimal values in the ratio columns (609 duplicates became 307). Plain duplicates were kept, because two different borrowers could match exactly.
   - Removed one row with an impossible age.
   - Treated the value 98 in the three late-payment columns as a code for unknown (it cannot be a real count), and flagged those rows.
3. **Train/test split.** 80% training and 20% test, stratified so both sets keep the same share of defaulters.
4. **Cleaning that learns from data (training set only).** Every number was calculated from the training set and then applied to both sets, to avoid data leakage.
   - Blank `MonthlyIncome` and `NumberOfDependents` were flagged and filled with the training median.
   - About 90% of the huge `DebtRatio` values had a blank income, so they are probably not real ratios. They were flagged and capped at [100].
   - `RevolvingUtilizationOfUnsecuredLines` values were flagged and capped at the training 99.5th percentile ([1.37]).
5. **Exploratory analysis (training set only).** Late-payment counts and credit utilization were the strongest signals against the target.
6. **Feature engineering.** Three new columns built from the late-payment counts: `total_late`, `any_late` and `serious_late`.
7. **Models.**
   - Logistic Regression on scaled features, with balanced class weights.
   - Random Forest ([200] trees, max depth [10], minimum [20] rows per leaf, balanced class weights).
8. **Evaluation.** Precision, Recall, F1-Score and ROC-AUC on the test set.

## Why not accuracy?

About 93% of borrowers do not default. A model that says "no default" for everyone would be 93% accurate and would catch no defaulters. So the models are judged by:

- **Recall:** of the borrowers who really defaulted, how many the model caught.
- **Precision:** of the borrowers the model flagged, how many really defaulted.
- **F1-Score:** one number that balances Precision and Recall.
- **ROC-AUC:** how well the model ranks real defaulters above non-defaulters.

Balanced class weights raise Recall but lower Precision, so the model flags many borrowers who did not default. This is the expected trade-off for this data.

## Repository structure

```
CodeAlpha_CreditScoring/
├── credit_scoring.ipynb   # full analysis, cleaning, models and results
├── README.md
├── requirements.txt
├── randomforestmodel.joblib           # saved Random Forest
└── data/                  # place cs-training.csv here
```

## How to run

1. Download `cs-training.csv` from the Kaggle link above and put it in the `data/` folder.
2. Install the libraries:
   ```
   pip install -r requirements.txt
   ```
3. Open `credit_scoring.ipynb` and choose Run All.

## Limitations

- The scores come from one train/test split, and the gap between the two models is small, so I would not claim that one is clearly better.
- Precision is low because the model flags generously to catch more defaulters.
- The model settings were chosen by reasoning and were not tuned.

## Future work

- Hyperparameter tuning of the Random Forest, using cross-validation on the training set
- Trying other models, such as gradient boosting
- Choosing the decision cutoff to suit the cost of a missed defaulter versus a false alarm
- Deploying the model with a simple web interface

## Author

Samuel Amadasun. Built for the CodeAlpha Machine Learning internship.
