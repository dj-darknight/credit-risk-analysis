Individual Task 1 — Part 1.3: Credit Risk Data Analysis
Analysis code for Part 1.3 of the Individual Task 1 assignment, anchored around a CommBank Senior Data Scientist – Credit Risk role (see Part 1.1).

Story
Do individual-level credit-default risk drivers generalise across different lending populations, or does predictive power depend on what data is actually available about each borrower?

Two real, publicly available datasets are used:

Give Me Some Credit (GMSC) — 150,000 established revolving-credit borrowers, source: https://www.kaggle.com/c/give-me-some-credit/data
Home Credit Default Risk (HCDR) — a working copy (215,257 rows, main application_train.csv fields) of applicants with thinner/no prior credit history, source: https://www.kaggle.com/competitions/home-credit-default-risk/data
Setup
pip install -r requirements.txt
Download the data (not committed to this repo due to size — see .gitignore):

mkdir -p data
# GMSC (150,000 rows, ~13MB)
curl -L -o cs-training.csv \
  "https://raw.githubusercontent.com/vivekkalyan/give-me-some-credit/master/cs-training.csv"

# HCDR (215,257 rows, ~102MB unzipped) — official Kaggle competition data
# also available as a working copy at:
curl -L -o train.csv.zip \
  "https://raw.githubusercontent.com/sultanbeishenkulov/home-credit-default-risk/main/train.csv.zip"
unzip train.csv.zip -d homecredit
Place cs-training.csv in the repo root and homecredit/train.csv in a homecredit/ subfolder, matching the paths in analysis.py.

Run
python analysis.py
Outputs full console evaluation and saves results.json with all metrics and feature importances used in the Part 1.3 write-up.

Method
Two algorithms, applied identically to both datasets: Decision Tree (max_depth=6, class-weight balanced) and kNN (k=15, distance-weighted, on standardised features). Metrics: ROC-AUC, precision, recall, F1, accuracy — selected because both datasets are heavily class-imbalanced (defaulters are a small minority), so accuracy alone is misleading.

Key finding
GMSC's behavioural features (credit utilisation, past-due history) substantially outperformed HCDR's demographic/employment features for predicting default — suggesting predictive power in credit risk modelling depends more on whether behavioural repayment history is available than on which borrower population is being modelled. Full discussion in the assignment report.

Part 2 — Deliberation (cross-validation, learning curves, fairness audit)
Extends the above pipeline with three diagnostics, run on the same data, preprocessing, and hyperparameters:

pip install fairlearn
python part2_analysis.py
5-fold stratified cross-validation for both models on both datasets.
Learning curves (train vs. validation ROC-AUC across 6 training-set sizes).
Fairlearn bias audit on the HCDR Decision Tree, using applicant age band as the sensitive attribute (demographic parity, equalized odds).
Note on kNN sample size: cross-validation and learning curves for kNN use an 8,000-row stratified subsample of each dataset, not the full 150k/215k rows. Distance-weighted kNN CV/learning-curve computation on the full datasets was not feasible in standard compute environments. Decision Tree steps use the full dataset in both cases. This is disclosed here and in the assignment report rather than applied silently.

Saves part2_results.json with all CV, learning-curve, and fairness metrics used in the Part 2 write-up. Expected runtime: under a minute.
