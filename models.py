"""
Predictive modelling: one binary classifier per banking service estimates
P(customer uses / is interested in the service | profile, activity, other products held).

Interest is modelled with *implicit feedback*: current product holding is the positive label.
For a target service p, the features are the preprocessed profile/behaviour features plus the
holding flags of the OTHER services (never the target itself -> no label leakage).

Candidates compared (selected per product by 5-fold CV ROC-AUC on the training split):
    Baseline (prior)      - predicts the training prevalence (no learning)
    LogisticRegression    - linear, interpretable
    RandomForest          - non-linear, bagging
    GradientBoosting      - non-linear, boosting (HistGradientBoostingClassifier)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

from data_generator import PRODUCTS


def make_features(X_base: np.ndarray, own: pd.DataFrame, target: str) -> np.ndarray:
    others = [p for p in PRODUCTS if p != target]
    return np.hstack([X_base, own[others].to_numpy(dtype=float)])


def feature_names_for(base_names: list[str], target: str) -> list[str]:
    return base_names + [f"owns_{p}" for p in PRODUCTS if p != target]


def candidate_models(seed: int = 42) -> dict:
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, C=1.0),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=5, n_jobs=-1, random_state=seed),
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, learning_rate=0.05, max_depth=3, random_state=seed),
    }


def train_product_models(X_tr, own_tr, X_te, own_te, seed: int = 42):
    """Train/evaluate all candidates for every product.

    Returns (results_df, best_models) where best_models[product] = (model_name, fitted_model).
    """
    rows, best = [], {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for p in PRODUCTS:
        Xp_tr, Xp_te = make_features(X_tr, own_tr, p), make_features(X_te, own_te, p)
        y_tr, y_te = own_tr[p].to_numpy(), own_te[p].to_numpy()

        models = {"Baseline (prior)": DummyClassifier(strategy="prior")}
        models.update(candidate_models(seed))
        fitted, cv_auc = {}, {}
        for name, model in models.items():
            cv_auc[name] = (np.nan if name.startswith("Baseline")
                            else cross_val_score(model, Xp_tr, y_tr, cv=cv, scoring="roc_auc").mean())
            model.fit(Xp_tr, y_tr)
            fitted[name] = model
            proba = model.predict_proba(Xp_te)[:, 1]
            pred = (proba >= 0.5).astype(int)
            rows.append({
                "product": p, "model": name,
                "cv_roc_auc": cv_auc[name],
                "test_roc_auc": roc_auc_score(y_te, proba),
                "test_pr_auc": average_precision_score(y_te, proba),
                "test_precision": precision_score(y_te, pred, zero_division=0),
                "test_recall": recall_score(y_te, pred, zero_division=0),
                "test_f1": f1_score(y_te, pred, zero_division=0),
                "prevalence": y_te.mean(),
            })
        chosen = max((n for n in cv_auc if not n.startswith("Baseline")), key=lambda n: cv_auc[n])
        best[p] = (chosen, fitted[chosen])

    results = pd.DataFrame(rows)
    results["selected"] = results.apply(lambda r: best[r["product"]][0] == r["model"], axis=1)
    return results, best


def predict_interest(best_models: dict, X_base: np.ndarray, own: pd.DataFrame) -> pd.DataFrame:
    """Predicted interest (probability) in every service -> DataFrame (n_customers x n_products)."""
    out = {}
    for p in PRODUCTS:
        _, model = best_models[p]
        out[p] = model.predict_proba(make_features(X_base, own, p))[:, 1]
    return pd.DataFrame(out, index=own.index)


def compute_importance(best_models, X_te, own_te, base_names, seed: int = 42) -> dict:
    """Permutation importance (drop in ROC-AUC) of the selected model for each product."""
    result = {}
    for p in PRODUCTS:
        _, model = best_models[p]
        Xp = make_features(X_te, own_te, p)
        pi = permutation_importance(model, Xp, own_te[p].to_numpy(), scoring="roc_auc",
                                    n_repeats=5, random_state=seed, n_jobs=1)
        result[p] = pd.Series(pi.importances_mean, index=feature_names_for(base_names, p)
                              ).sort_values(ascending=False)
    return result
