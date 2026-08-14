"""
Part 1.3: Data Analysis
CommBank Senior Data Scientist - Credit Risk (Indivdiual Project)

Two datasets, one story: do individual-level credit-default risk drivers
generalise across different lending populations?
  Dataset A: Give Me Some Credit (GMSC) - established revolving-credit borrowers
  Dataset B: Home Credit Default Risk (HCDR) - broader/thin-credit-history applicants

Same two algorithms on both: Decision Tree, kNN (both differ from the
item-based collaborative filtering used in Practical Data Science last semester).
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score,confusion_matrix, accuracy_score)

RANDOM_STATE = 4175665  # My student ID, for reproducibility

results = {}

# ----------------------------------------------------------------------
# DATASET A: Give Me Some Credit
# ----------------------------------------------------------------------
print("=" * 70)
print("DATASET A: Give Me Some Credit (GMSC)")
print("=" * 70)

gmsc = pd.read_csv("cs-training.csv")
gmsc = gmsc.drop(columns=["Id"])

# Impute missing values (median for income/dependents - robust to outliers)
imputer_gmsc = SimpleImputer(strategy="median")
gmsc_imputed = pd.DataFrame(imputer_gmsc.fit_transform(gmsc), columns=gmsc.columns)

y_a = gmsc_imputed["SeriousDlqin2yrs"].astype(int)
X_a = gmsc_imputed.drop(columns=["SeriousDlqin2yrs"])

print(f"Shape: {X_a.shape}, Target positive rate: {y_a.mean():.4f}")

Xa_train, Xa_test, ya_train, ya_test = train_test_split(X_a, y_a, test_size=0.2, random_state=RANDOM_STATE, stratify=y_a)

scaler_a = StandardScaler()
Xa_train_scaled = scaler_a.fit_transform(Xa_train)
Xa_test_scaled = scaler_a.transform(Xa_test)

# Decision Tree (depth-limited to avoid overfitting given class imbalance)
dt_a = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)
dt_a.fit(Xa_train, ya_train)
dt_a_proba = dt_a.predict_proba(Xa_test)[:, 1]
dt_a_pred = dt_a.predict(Xa_test)

# kNN (on scaled features - distance-based, needs scaling)
knn_a = KNeighborsClassifier(n_neighbors=15, weights="distance")
knn_a.fit(Xa_train_scaled, ya_train)
knn_a_proba = knn_a.predict_proba(Xa_test_scaled)[:, 1]
knn_a_pred = knn_a.predict(Xa_test_scaled)


def eval_model(y_true, y_pred, y_proba, name):
    metrics = { "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
                "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
                "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
                "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
                "accuracy": round(accuracy_score(y_true, y_pred), 4), }
    cm = confusion_matrix(y_true, y_pred).tolist()
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  confusion_matrix [[TN,FP],[FN,TP]]: {cm}")
    metrics["confusion_matrix"] = cm
    return metrics


results["gmsc_decision_tree"] = eval_model(ya_test, dt_a_pred, dt_a_proba, "GMSC - Decision Tree")
results["gmsc_knn"] = eval_model(ya_test, knn_a_pred, knn_a_proba, "GMSC - kNN (k=15)")

# Feature importance for Decision Tree
gmsc_importance = pd.Series(dt_a.feature_importances_, index=X_a.columns).sort_values(ascending=False)
print("\nGMSC Decision Tree - top feature importances:")
print(gmsc_importance.head(6))
results["gmsc_feature_importance"] = gmsc_importance.round(4).to_dict()

# ----------------------------------------------------------------------
# DATASET B: Home Credit Default Risk
# ----------------------------------------------------------------------
print("\n" + "=" * 70)
print("DATASET B: Home Credit Default Risk (HCDR)")
print("=" * 70)

hcdr_raw = pd.read_csv("homecredit/train.csv")

# Select a comparable, well-populated feature set (avoid the >50%-missing
# building/apartment columns identified during inspection)
feature_cols = [
    "AMT_INCOME_TOTAL",     # ~ income (like GMSC MonthlyIncome)
    "AMT_CREDIT",           # loan amount requested
    "AMT_ANNUITY",          # loan repayment burden
    "DAYS_AGE",             # ~ age
    "DAYS_EMPLOYMENT",      # employment tenure
    "CNT_CHILDREN",         # ~ dependents
    "CNT_FAM_MEMBERS",
    "REGION_RATING_CLIENT", # region risk rating
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "NAME_INCOME_TYPE",
]
hcdr = hcdr_raw[feature_cols + ["TARGET"]].copy()

# DAYS_EMPLOYMENT has a known Home Credit data-quality anomaly: 365243 = "not employed" placeholder
hcdr["DAYS_EMPLOYMENT"] = hcdr["DAYS_EMPLOYMENT"].replace(365243, np.nan)
# Convert day-counts (negative, days before application) into positive years for interpretability
hcdr["AGE_YEARS"] = -hcdr["DAYS_AGE"] / 365
hcdr["EMPLOYMENT_YEARS"] = -hcdr["DAYS_EMPLOYMENT"] / 365
hcdr = hcdr.drop(columns=["DAYS_AGE", "DAYS_EMPLOYMENT"])

# One-hot encode categoricals
cat_cols = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_EDUCATION_TYPE",
            "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_INCOME_TYPE"]
hcdr_encoded = pd.get_dummies(hcdr, columns=cat_cols, drop_first=True)

y_b = hcdr_encoded["TARGET"].astype(int)
X_b = hcdr_encoded.drop(columns=["TARGET"])

# Impute remaining numeric NaNs (median)
imputer_b = SimpleImputer(strategy="median")
X_b = pd.DataFrame(imputer_b.fit_transform(X_b), columns=X_b.columns)

print(f"Shape: {X_b.shape}, Target positive rate: {y_b.mean():.4f}")

Xb_train, Xb_test, yb_train, yb_test = train_test_split( X_b, y_b, test_size=0.2, random_state=RANDOM_STATE, stratify=y_b)

scaler_b = StandardScaler()
Xb_train_scaled = scaler_b.fit_transform(Xb_train)
Xb_test_scaled = scaler_b.transform(Xb_test)

dt_b = DecisionTreeClassifier(
    max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
)
dt_b.fit(Xb_train, yb_train)
dt_b_proba = dt_b.predict_proba(Xb_test)[:, 1]
dt_b_pred = dt_b.predict(Xb_test)

knn_b = KNeighborsClassifier(n_neighbors=15, weights="distance")
knn_b.fit(Xb_train_scaled, yb_train)
knn_b_proba = knn_b.predict_proba(Xb_test_scaled)[:, 1]
knn_b_pred = knn_b.predict(Xb_test_scaled)

results["hcdr_decision_tree"] = eval_model(yb_test, dt_b_pred, dt_b_proba, "HCDR - Decision Tree")
results["hcdr_knn"] = eval_model(yb_test, knn_b_pred, knn_b_proba, "HCDR - kNN (k=15)")

hcdr_importance = pd.Series(dt_b.feature_importances_, index=X_b.columns).sort_values(ascending=False)
print("\nHCDR Decision Tree - top feature importances:")
print(hcdr_importance.head(8))
results["hcdr_feature_importance"] = hcdr_importance.round(4).to_dict()

# ----------------------------------------------------------------------
# Save results for reporting
# ----------------------------------------------------------------------
with open("results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nDone. Results saved to results.json")
