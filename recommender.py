"""
Recommendation engine + offline evaluation.

Methods compared
  Random                      - sanity-check lower bound
  Popularity                  - most commonly held services first (non-personalised baseline)
  Item-CF                     - item-based collaborative filtering (cosine similarity of
                                product co-holding: "customers holding A also hold B")
  Model (predicted interest)  - per-product classifiers from models.py (profile + activity)
  Hybrid                      - alpha * model score + (1 - alpha) * item-CF score
                                (row-wise min-max scaled)

Evaluation protocol: leave-one-out on held-out TEST customers who hold >= 2 services.
One held service is hidden at random; all methods rank the services the customer does not
(visibly) hold; we measure where the hidden service lands.
Metrics: HitRate@K, MRR, NDCG@3, Coverage@1 (share of catalogue ever recommended first).
Ties are handled by expected rank under random tie-breaking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data_generator import PRODUCT_LABELS

# Feedback topic -> products whose cross-sell should be handled carefully
AREA_TO_PRODUCTS = {
    "loan_service": ["Personal_Loan", "Home_Loan"],
    "credit_card": ["Credit_Card"],
    "interest_rates": ["Fixed_Deposit"],
    "mobile_app": ["Digital_Banking_Plus"],
    "upi_payments": ["Digital_Banking_Plus"],
}


def _rowwise_minmax(a: np.ndarray) -> np.ndarray:
    lo, hi = a.min(1, keepdims=True), a.max(1, keepdims=True)
    return (a - lo) / np.where(hi - lo == 0, 1, hi - lo)


class HybridRecommender:
    def __init__(self, products: list[str], alpha: float = 0.7):
        self.products = list(products)
        self.alpha = alpha

    def fit(self, own_train: pd.DataFrame):
        M = own_train[self.products].to_numpy(float)
        self.popularity_ = M.mean(0)
        norms = np.linalg.norm(M, axis=0)
        norms[norms == 0] = 1.0
        sim = (M.T @ M) / np.outer(norms, norms)
        np.fill_diagonal(sim, 0.0)
        self.sim_ = sim
        return self

    # ---- scoring -------------------------------------------------------- #
    def score_popularity(self, own: pd.DataFrame) -> np.ndarray:
        return np.tile(self.popularity_, (len(own), 1))

    def score_itemcf(self, own: pd.DataFrame) -> np.ndarray:
        return own[self.products].to_numpy(float) @ self.sim_

    def score_hybrid(self, probs: np.ndarray, own: pd.DataFrame) -> np.ndarray:
        return (self.alpha * _rowwise_minmax(np.asarray(probs, float))
                + (1 - self.alpha) * _rowwise_minmax(self.score_itemcf(own)))

    # ---- recommendations with explanations ------------------------------ #
    def recommend_batch(self, own: pd.DataFrame, probs: pd.DataFrame, k: int = 3,
                        neg_areas: dict | None = None) -> pd.DataFrame:
        own_m = own[self.products].to_numpy(float)
        P = probs[self.products].to_numpy(float)
        H = np.where(own_m == 1, -np.inf, self.score_hybrid(P, own))
        rows = []
        for r in range(len(own)):
            cid = own.index[r]
            areas = (neg_areas or {}).get(cid, set())
            owned_idx = np.flatnonzero(own_m[r])
            for rank, j in enumerate(np.argsort(-H[r])[:k], start=1):
                if not np.isfinite(H[r, j]):
                    break
                p = self.products[j]
                reason = f"{P[r, j]:.0%} predicted interest"
                if len(owned_idx):
                    i = owned_idx[np.argmax(self.sim_[owned_idx, j])]
                    if self.sim_[i, j] > 0.05:
                        reason += f"; often held together with {PRODUCT_LABELS[self.products[i]]}"
                caution = ""
                hit = [a for a in areas if p in AREA_TO_PRODUCTS.get(a, [])]
                if hit:
                    caution = ("Customer gave negative feedback on "
                               + ", ".join(sorted(h.replace("_", " ") for h in hit))
                               + " - resolve the service issue before cross-selling")
                rows.append({"customer_id": cid, "rank": rank, "product": p,
                             "product_name": PRODUCT_LABELS[p],
                             "interest_prob": round(float(P[r, j]), 4),
                             "hybrid_score": round(float(H[r, j]), 4),
                             "reason": reason, "caution": caution})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Offline evaluation
# --------------------------------------------------------------------------- #
def leave_one_out_split(own: pd.DataFrame, products: list[str], seed: int = 42):
    """Hide one held service per eligible customer (>= 2 holdings)."""
    rng = np.random.default_rng(seed)
    M = own[products].to_numpy(int)
    eligible = np.flatnonzero(M.sum(1) >= 2)
    hidden = np.array([rng.choice(np.flatnonzero(M[i])) for i in eligible])
    masked = M[eligible].copy()
    masked[np.arange(len(eligible)), hidden] = 0
    return eligible, masked, hidden


def _expected(gains: np.ndarray, greater: np.ndarray, ties: np.ndarray) -> np.ndarray:
    cum = np.concatenate([[0.0], np.cumsum(gains)])
    return (cum[greater + ties] - cum[greater]) / ties


def ranking_metrics(scores: np.ndarray, masked: np.ndarray, hidden: np.ndarray,
                    ks=(1, 2, 3)) -> dict:
    s = np.asarray(scores, float).copy()
    s[masked == 1] = -np.inf                      # already-held services are not candidates
    m = s.shape[1]
    hs = s[np.arange(len(s)), hidden]
    greater = (s > hs[:, None]).sum(1)
    ties = (s == hs[:, None]).sum(1)              # includes the hidden item itself
    ranks = np.arange(1, m + 1)
    out = {f"HitRate@{k}": _expected((ranks <= k).astype(float), greater, ties).mean() for k in ks}
    out["MRR"] = _expected(1.0 / ranks, greater, ties).mean()
    out["NDCG@3"] = _expected(np.where(ranks <= 3, 1 / np.log2(ranks + 1), 0.0), greater, ties).mean()
    out["Coverage@1"] = len(np.unique(s.argmax(1))) / m
    return out
