"""
Data cleaning + feature engineering + sklearn preprocessing pipeline.

Cleaning (deterministic rules, no statistics learned from data -> no leakage):
  * drop duplicate customer records
  * standardise inconsistent labels (gender: "M"/"female " -> Male/Female, city tier spelling)
  * impossible values -> NaN (age outside 18-100, income <= 0, credit score outside 300-900 ...)
  * business-rule outlier cap: monthly income > 2,000,000 treated as a data-entry error -> NaN
  * missing categoricals -> "Unknown" category

Learned steps (median imputation, scaling, one-hot) live in a sklearn ColumnTransformer
that is fitted on the TRAINING split only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_RAW = [
    "age", "tenure_years", "income", "credit_score", "avg_balance", "monthly_spend",
    "savings_rate", "monthly_txn_count", "upi_txn_count", "card_txn_count",
    "atm_withdrawals", "branch_visits", "login_count", "mobile_app_user",
    "existing_emi_ratio", "num_dependents", "has_salary_account",
]
NUMERIC_ENGINEERED = [
    "income_log", "balance_log", "spend_to_income", "balance_to_income_months",
    "digital_txn_share", "branch_dependence", "high_emi_flag", "is_new_customer",
]
CATEGORICAL = ["gender", "occupation", "city_tier", "age_band", "credit_band"]
NUMERIC = NUMERIC_RAW + NUMERIC_ENGINEERED
ALL_FEATURES = NUMERIC + CATEGORICAL


def clean_customers(raw: pd.DataFrame):
    """Return (clean_df, report_dict)."""
    report: dict = {"rows_raw": int(len(raw))}
    df = raw.copy()
    report["missing_values_raw"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}

    before = len(df)
    df = df.drop_duplicates(subset="customer_id", keep="first").reset_index(drop=True)
    report["duplicate_rows_removed"] = int(before - len(df))

    # --- categorical standardisation -------------------------------------- #
    report["gender_labels_standardised"] = int(
        (df["gender"].notna() & ~df["gender"].isin(["Male", "Female", "Other"])).sum())
    g = df["gender"].astype("string").str.strip().str.lower()
    g = g.map({"m": "Male", "male": "Male", "f": "Female", "female": "Female",
               "o": "Other", "other": "Other"})
    df["gender"] = g.astype("object").fillna("Unknown")

    report["city_tier_labels_standardised"] = int(
        (df["city_tier"].notna() & ~df["city_tier"].isin(["Tier-1", "Tier-2", "Tier-3"])).sum())
    key = df["city_tier"].astype("string").str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
    df["city_tier"] = key.map({"tier1": "Tier-1", "tier2": "Tier-2", "tier3": "Tier-3"}) \
        .astype("object").fillna("Unknown")

    df["occupation"] = df["occupation"].astype("string").str.strip().astype("object").fillna("Unknown")

    # --- numeric validity rules -------------------------------------------- #
    for c in NUMERIC_RAW:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    rules = {
        "invalid_age": (df["age"] < 18) | (df["age"] > 100),
        "non_positive_income": df["income"] <= 0,
        "income_outlier_over_2M": df["income"] > 2_000_000,
        "invalid_credit_score": (df["credit_score"] < 300) | (df["credit_score"] > 900),
        "negative_balance": df["avg_balance"] < 0,
        "invalid_savings_rate": (df["savings_rate"] < 0) | (df["savings_rate"] > 1),
    }
    cols = {"invalid_age": "age", "non_positive_income": "income",
            "income_outlier_over_2M": "income", "invalid_credit_score": "credit_score",
            "negative_balance": "avg_balance", "invalid_savings_rate": "savings_rate"}
    for name, mask in rules.items():
        report[name] = int(mask.fillna(False).sum())
        df.loc[mask.fillna(False), cols[name]] = np.nan

    report["rows_clean"] = int(len(df))
    report["missing_values_after_cleaning"] = {
        c: int(v) for c, v in df[NUMERIC_RAW].isna().sum().items() if v > 0}
    return df, report


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add behavioural / financial-health features."""
    d = df.copy()
    inc = d["income"].where(d["income"] > 0)

    d["income_log"] = np.log1p(d["income"])
    d["balance_log"] = np.log1p(d["avg_balance"])
    d["spend_to_income"] = d["monthly_spend"] / inc                    # spending pressure
    d["balance_to_income_months"] = d["avg_balance"] / inc             # liquidity cushion
    d["digital_txn_share"] = d["upi_txn_count"] / (
        d["upi_txn_count"] + d["card_txn_count"] + d["atm_withdrawals"] + 1)
    d["branch_dependence"] = d["branch_visits"] / (d["branch_visits"] + d["login_count"] + 1)
    d["high_emi_flag"] = (d["existing_emi_ratio"] > 0.4).astype(int)
    d["is_new_customer"] = (d["tenure_years"] < 1).astype(int)

    d["age_band"] = pd.cut(d["age"], [0, 25, 35, 45, 55, 200],
                           labels=["<=25", "26-35", "36-45", "46-55", "56+"]
                           ).astype("object").fillna("Unknown")
    d["credit_band"] = pd.cut(d["credit_score"], [0, 600, 700, 750, 901],
                              labels=["poor", "fair", "good", "excellent"]
                              ).astype("object").fillna("Unknown")

    d[NUMERIC_ENGINEERED] = d[NUMERIC_ENGINEERED].replace([np.inf, -np.inf], np.nan)
    return d


def build_preprocessor() -> ColumnTransformer:
    """Median-impute (+ missing indicators) and scale numerics; one-hot categoricals."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
    ])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
        sparse_threshold=0.0,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [n.split("__", 1)[-1] for n in preprocessor.get_feature_names_out()]
