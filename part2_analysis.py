"""
Part 2 deliberation: 5-fold cross-validation, learning curves, and a
Fairlearn bias audit, extending Task 1's analysis.py pipeline.

Notes:
- Uses the same preprocessing, hyperparameters, and random_state as
  analysis.py.
- kNN cross-validation and learning curves use an 8,000-row stratified
  subsample of each dataset. Distance-weighted kNN CV/learning-curve
  computation on the full 150k/215k-row datasets was not feasible in
  standard compute environments; this subsample is disclosed here and
  in the report rather than silently applied. Decision Tree steps use
  the full dataset in both cases.
- Checkpoints to part2_results.json after each stage so partial
  progress is never lost on a long run.

Requires: pandas, numpy, scikit-learn, fairlearn
Run: python3 part2_analysis.py
Expected runtime: a few minutes (kNN steps are the slow part).
"""
import pandas as pd
import numpy as np
import json
import time
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from fairlearn.metrics import MetricFrame, demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference
from sklearn.metrics import recall_score, precision_score

RANDOM_STATE = 4175665
SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
SCORING = ["roc_auc", "precision", "recall", "f1"]
KNN_SUBSAMPLE_N = 8000
TRAIN_SIZES = np.linspace(0.1, 1.0, 6)

results = {}


def log(*a):
    print(*a, flush=True)


def checkpoint():
    with open("part2_results.json", "w") as f:
        json.dump(results, f, indent=2)


def load_gmsc(path="cs-training.csv"):
    df = pd.read_csv(path).drop(columns=["Id"])
    imputer = SimpleImputer(strategy="median")
    df_imp = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    y = df_imp["SeriousDlqin2yrs"].astype(int)
    X = df_imp.drop(columns=["SeriousDlqin2yrs"])
    return X, y


def load_hcdr(path="homecredit/train.csv"):
    raw = pd.read_csv(path)
    feature_cols = ["AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY", "DAYS_AGE", "DAYS_EMPLOYMENT",
                    "CNT_CHILDREN", "CNT_FAM_MEMBERS", "REGION_RATING_CLIENT", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
                    "NAME_EDUCATION_TYPE", "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_INCOME_TYPE"]
    df = raw[feature_cols + ["TARGET"]].copy()
    df["DAYS_EMPLOYMENT"] = df["DAYS_EMPLOYMENT"].replace(365243, np.nan)
    df["AGE_YEARS"] = -df["DAYS_AGE"] / 365
    df["EMPLOYMENT_YEARS"] = -df["DAYS_EMPLOYMENT"] / 365
    df = df.drop(columns=["DAYS_AGE", "DAYS_EMPLOYMENT"])
    cat_cols = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "NAME_EDUCATION_TYPE",
                "NAME_FAMILY_STATUS", "NAME_HOUSING_TYPE", "NAME_INCOME_TYPE"]
    encoded = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    y = encoded["TARGET"].astype(int)
    X_raw = encoded.drop(columns=["TARGET"])
    imputer = SimpleImputer(strategy="median")
    X = pd.DataFrame(imputer.fit_transform(X_raw), columns=X_raw.columns)
    age = df[["AGE_YEARS"]].copy()
    return X, y, age


def stratified_subsample(X, y, n, seed=RANDOM_STATE):
    idx, _ = train_test_split(np.arange(len(y)), train_size=min(n, len(y)), stratify=y, random_state=seed)
    return X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)


def cv_metrics(estimator, X, y):
    cv = cross_validate(estimator, X, y, cv=SKF, scoring=SCORING, n_jobs=2)
    return {m: {"mean": round(cv[f"test_{m}"].mean(), 4), "std": round(cv[f"test_{m}"].std(), 4)} for m in SCORING}


def lc_metrics(estimator, X, y):
    sizes, tr, va = learning_curve(estimator, X, y, cv=SKF, scoring="roc_auc",
                                    train_sizes=TRAIN_SIZES, n_jobs=2, random_state=RANDOM_STATE)
    return {"train_sizes": sizes.tolist(),
            "train_auc_mean": tr.mean(axis=1).round(4).tolist(),
            "val_auc_mean": va.mean(axis=1).round(4).tolist(),
            "val_auc_std": va.std(axis=1).round(4).tolist()}


def make_dt():
    return DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE)


def make_knn():
    return Pipeline([("scale", StandardScaler()), ("knn", KNeighborsClassifier(n_neighbors=15, weights="distance"))])


def main():
    t0 = time.time()

    log("Loading data...")
    X_a, y_a = load_gmsc()
    X_b, y_b, age_b = load_hcdr()
    log(f"GMSC: {X_a.shape}, HCDR: {X_b.shape}  ({time.time()-t0:.1f}s)")

    X_a_s, y_a_s = stratified_subsample(X_a, y_a, KNN_SUBSAMPLE_N)
    X_b_s, y_b_s = stratified_subsample(X_b, y_b, KNN_SUBSAMPLE_N)

    # 1. Cross-validation
    log("Running 5-fold CV...")
    results["gmsc_cv"] = {"decision_tree": cv_metrics(make_dt(), X_a, y_a),
                           "knn": cv_metrics(make_knn(), X_a_s, y_a_s)}
    results["hcdr_cv"] = {"decision_tree": cv_metrics(make_dt(), X_b, y_b),
                           "knn": cv_metrics(make_knn(), X_b_s, y_b_s)}
    checkpoint()
    log(f"CV done ({time.time()-t0:.1f}s)")

    # 2. Learning curves
    log("Running learning curves...")
    results["gmsc_lc_dt"] = lc_metrics(make_dt(), X_a, y_a)
    checkpoint()
    results["hcdr_lc_dt"] = lc_metrics(make_dt(), X_b, y_b)
    checkpoint()
    results["gmsc_lc_knn"] = lc_metrics(make_knn(), X_a_s, y_a_s)
    checkpoint()
    results["hcdr_lc_knn"] = lc_metrics(make_knn(), X_b_s, y_b_s)
    checkpoint()
    log(f"Learning curves done ({time.time()-t0:.1f}s)")

    # 3. Fairness audit (HCDR, age bands)
    log("Running fairness audit...")
    Xb_train, Xb_test, yb_train, yb_test, age_train, age_test = train_test_split(
        X_b, y_b, age_b, test_size=0.2, random_state=RANDOM_STATE, stratify=y_b)
    dt = make_dt()
    dt.fit(Xb_train, yb_train)
    pred = dt.predict(Xb_test)
    age_group = pd.cut(age_test["AGE_YEARS"], bins=[0, 30, 45, 60, 100],
                        labels=["<30", "30-45", "45-60", "60+"]).astype(str)

    by_group = {}
    for grp in ["<30", "30-45", "45-60", "60+"]:
        mask = age_group == grp
        sub_true, sub_pred = yb_test[mask.values], pred[mask.values]
        by_group[grp] = {
            "n": int(mask.sum()),
            "actual_default_rate": round(float(sub_true.mean()), 4),
            "flagged_rate": round(float(sub_pred.mean()), 4),
            "recall_within_group": round(float(sub_pred[sub_true == 1].mean()), 4) if sub_true.sum() > 0 else None,
        }
    results["hcdr_fairness_by_age"] = by_group

    mf = MetricFrame(metrics={"selection_rate": lambda yt, yp: yp.mean(),
                               "recall": recall_score, "precision": precision_score},
                      y_true=yb_test, y_pred=pred, sensitive_features=age_group)
    results["hcdr_fairlearn"] = {
        "by_group": mf.by_group.round(4).to_dict(),
        "demographic_parity_difference": round(float(demographic_parity_difference(yb_test, pred, sensitive_features=age_group)), 4),
        "demographic_parity_ratio": round(float(demographic_parity_ratio(yb_test, pred, sensitive_features=age_group)), 4),
        "equalized_odds_difference": round(float(equalized_odds_difference(yb_test, pred, sensitive_features=age_group)), 4),
    }
    checkpoint()
    log(f"Fairness audit done ({time.time()-t0:.1f}s)")
    log("All results saved to part2_results.json")


if __name__ == "__main__":
    main()
