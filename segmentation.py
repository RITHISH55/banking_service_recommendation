"""Customer segmentation (K-Means) for marketing analysis / customer insights."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEG_FEATURES = ["age", "income_log", "balance_log", "credit_score", "savings_rate",
                "digital_txn_share", "login_count", "branch_dependence"]

TRAITS = {
    "age": ("older", "younger"),
    "income_log": ("high-income", "lower-income"),
    "balance_log": ("high-balance", "low-balance"),
    "credit_score": ("strong-credit", "weaker-credit"),
    "savings_rate": ("high-saver", "low-saver"),
    "digital_txn_share": ("digital-first", "cash/card-led"),
    "login_count": ("app-active", "app-light"),
    "branch_dependence": ("branch-reliant", "branch-light"),
}


def segment_customers(df_feat: pd.DataFrame, k_range=range(3, 7), seed: int = 42):
    prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
    X = prep.fit_transform(df_feat[SEG_FEATURES])

    scores = {}
    for k in k_range:
        labels = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
        scores[k] = silhouette_score(X, labels, sample_size=min(2000, len(X)), random_state=seed)
    best_k = max(scores, key=scores.get)
    km = KMeans(best_k, n_init=10, random_state=seed).fit(X)
    labels = pd.Series(km.labels_, index=df_feat.index, name="segment")

    profile = df_feat[SEG_FEATURES].groupby(labels).mean()
    z = (profile - profile.mean()) / (profile.std() + 1e-9)
    names = {}
    for seg, row in z.iterrows():
        top = row.abs().sort_values(ascending=False).index[:2]
        words = [TRAITS[t][0] if row[t] > 0 else TRAITS[t][1] for t in top]
        names[seg] = f"S{seg}: {' / '.join(words)}"
    coords = PCA(2, random_state=seed).fit_transform(X)
    return {"labels": labels, "names": names, "profile": profile,
            "pca": pd.DataFrame(coords, index=df_feat.index, columns=["pc1", "pc2"]),
            "silhouette": scores, "best_k": best_k}
