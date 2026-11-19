"""Shared inference helpers used by demo.py (command line) and app.py (Streamlit)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from data_generator import PRODUCTS, PRODUCT_LABELS
from models import predict_interest
from preprocessing import ALL_FEATURES, NUMERIC_RAW, clean_customers, engineer_features
from sentiment import tag_topic

BUNDLE_PATH = Path(__file__).parent / "models" / "bundle.joblib"

RAW_COLUMNS = ["customer_id", "age", "gender", "occupation", "city_tier"] + [
    c for c in NUMERIC_RAW if c != "age"]

# An example of a brand-new customer (any field can be left out -> treated as missing & imputed)
EXAMPLE_PROFILE = {
    "age": 29, "gender": "Female", "occupation": "Salaried", "city_tier": "Tier-1",
    "tenure_years": 2.5, "income": 85000, "credit_score": 742, "avg_balance": 160000,
    "monthly_spend": 52000, "savings_rate": 0.22, "monthly_txn_count": 48,
    "upi_txn_count": 32, "card_txn_count": 9, "atm_withdrawals": 1, "branch_visits": 0,
    "login_count": 26, "mobile_app_user": 1, "existing_emi_ratio": 0.05,
    "num_dependents": 0, "has_salary_account": 1,
}


def load_bundle(path: Path = BUNDLE_PATH) -> dict:
    if not Path(path).exists():
        raise FileNotFoundError("models/bundle.joblib not found - run `python main.py` first.")
    return joblib.load(path)


def profile_to_features(profile: dict) -> pd.DataFrame:
    """Raw profile dict -> cleaned + engineered one-row feature frame."""
    row = {c: np.nan for c in RAW_COLUMNS}
    row.update(profile)
    row["customer_id"] = row.get("customer_id") if isinstance(row.get("customer_id"), str) else "NEW"
    raw = pd.DataFrame([row])
    for c in ("gender", "occupation", "city_tier"):
        raw[c] = raw[c].astype("object")
    clean, _ = clean_customers(raw)
    return engineer_features(clean).set_index("customer_id")


def score_features(bundle: dict, feat: pd.DataFrame, owned: list[str], k: int = 3,
                   neg_areas: set | None = None):
    """Predicted interest + top-k recommendations for a one-row feature frame."""
    X = bundle["preprocessor"].transform(feat[ALL_FEATURES])
    own = pd.DataFrame([[int(p in owned) for p in PRODUCTS]], columns=PRODUCTS, index=feat.index)
    probs = predict_interest(bundle["models"], X, own)
    cid = feat.index[0]
    recs = bundle["recommender"].recommend_batch(
        own, probs, k=k, neg_areas={cid: neg_areas} if neg_areas else None)
    return probs.iloc[0], recs


def score_existing_customer(bundle: dict, customer_id: str, k: int = 3):
    feat = bundle["customers"].loc[[customer_id]]
    own_row = bundle["ownership"].loc[customer_id]
    owned = [p for p in PRODUCTS if own_row[p] == 1]
    probs, recs = score_features(bundle, feat, owned, k, bundle["neg_areas"].get(customer_id))
    return feat.iloc[0], owned, probs, recs


def analyse_feedback_text(bundle: dict, text: str) -> dict:
    pipe = bundle["sentiment_pipe"]
    proba = dict(zip(pipe.classes_, pipe.predict_proba([text])[0]))
    return {"sentiment": pipe.predict([text])[0], "probabilities": proba, "topic": tag_topic(text)}


def label(p: str) -> str:
    return PRODUCT_LABELS[p]
