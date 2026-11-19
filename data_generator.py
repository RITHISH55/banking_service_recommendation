"""
Synthetic data generator for the Personalized Banking Service Recommendation project.

No public dataset contains customer profiles + product holdings + free-text feedback
for the same bank customers, so we create a *documented synthetic dataset*.

Three tables are produced (see README "Dataset" section):
  1. customers_raw.csv  - profile + account-activity features (deliberately "dirty":
                          missing values, inconsistent labels, impossible values, duplicates)
  2. product_usage.csv  - which of 6 banking services each customer currently uses (0/1)
  3. feedback.csv       - free-text customer feedback with rating and sentiment label

Product usage is generated from a hidden logistic model driven by customer features
(plus a shared "affluence" latent factor), so that real, learnable signal exists and
products are correlated with each other (which is what a recommender exploits).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRODUCTS = [
    "Fixed_Deposit",
    "Credit_Card",
    "Personal_Loan",
    "Home_Loan",
    "Mutual_Fund_SIP",
    "Digital_Banking_Plus",
]

PRODUCT_LABELS = {
    "Fixed_Deposit": "Fixed Deposit",
    "Credit_Card": "Credit Card",
    "Personal_Loan": "Personal Loan",
    "Home_Loan": "Home Loan",
    "Mutual_Fund_SIP": "Mutual Fund SIP",
    "Digital_Banking_Plus": "Digital Banking Plus (premium app features)",
}

TOPICS = [
    "mobile_app",
    "loan_service",
    "credit_card",
    "customer_support",
    "fees_charges",
    "interest_rates",
    "branch_service",
    "upi_payments",
]

# (negative, neutral, positive) probabilities per topic -> some areas are genuinely weaker
TOPIC_SENT_PROBS = {
    "mobile_app": (0.30, 0.20, 0.50),
    "loan_service": (0.50, 0.20, 0.30),
    "credit_card": (0.35, 0.20, 0.45),
    "customer_support": (0.50, 0.20, 0.30),
    "fees_charges": (0.55, 0.20, 0.25),
    "interest_rates": (0.40, 0.25, 0.35),
    "branch_service": (0.30, 0.25, 0.45),
    "upi_payments": (0.35, 0.15, 0.50),
}

TEMPLATES = {
    "mobile_app": {
        "positive": [
            "The mobile app is smooth and easy to use",
            "Love the new app design, transfers take only seconds",
            "The banking app is fast and reliable for daily use",
            "App navigation is simple and the dashboard is very clear",
        ],
        "negative": [
            "The mobile app keeps crashing during fund transfers",
            "App login is painfully slow and often fails",
            "Too many glitches in the banking app after the last release",
            "The app freezes every time I try to view my statement",
        ],
        "neutral": [
            "The app works okay but nothing special",
            "The mobile app is fine, some features are hard to find",
            "App does the basic job, an average experience",
        ],
    },
    "loan_service": {
        "positive": [
            "Loan approval was quick and the officer was very helpful",
            "Got my personal loan disbursed within two days",
            "The home loan process was transparent and well explained",
            "Smooth loan application with minimal paperwork",
        ],
        "negative": [
            "Loan processing took weeks and nobody kept me informed",
            "Too much paperwork for the loan and repeated document requests",
            "Loan approval was delayed without any explanation",
            "Home loan disbursement was slow and the process was confusing",
        ],
        "neutral": [
            "The loan process was average and took about a week",
            "Loan approval was okay, nothing exceptional",
            "Loan formalities were standard and took the expected time",
        ],
    },
    "credit_card": {
        "positive": [
            "Credit card rewards are great and the cashback is generous",
            "Happy with my credit card limit and reward points",
            "The credit card offers excellent travel benefits",
        ],
        "negative": [
            "Credit card annual fee is too high for the benefits",
            "My credit card was blocked without any notice and support was unhelpful",
            "Reward points on the credit card keep expiring too soon",
            "Credit card billing errors are never fixed on time",
        ],
        "neutral": [
            "The credit card is okay, rewards are average",
            "Credit card works fine but the benefits are standard",
        ],
    },
    "customer_support": {
        "positive": [
            "Customer care resolved my issue in a single call",
            "The support team was polite and very responsive",
            "Excellent support, the agent solved my problem quickly",
        ],
        "negative": [
            "Customer support never picks up the phone",
            "Waited forty minutes on hold and the issue is still unresolved",
            "Support agents gave me different answers every time I called",
            "Customer care is rude and unhelpful",
        ],
        "neutral": [
            "Support was okay, the issue got resolved after a follow up",
            "Customer care responded in a reasonable time",
        ],
    },
    "fees_charges": {
        "positive": [
            "No hidden charges and the fee structure is clear",
            "Happy that my account has zero maintenance fee",
            "Charges are reasonable and clearly explained",
        ],
        "negative": [
            "Unexpected charges were deducted from my account",
            "Too many hidden fees on small transactions",
            "The penalty for minimum balance is unfair",
            "Service charges keep increasing without any notice",
        ],
        "neutral": [
            "Fees are comparable to other banks",
            "Charges are standard, nothing surprising",
        ],
    },
    "interest_rates": {
        "positive": [
            "Fixed deposit rates are very attractive",
            "Good interest rate on my savings account",
            "Competitive interest rates compared to other banks",
        ],
        "negative": [
            "Savings interest rate is much lower than other banks",
            "Loan interest rates are too high",
            "FD rates are not competitive at all",
            "Interest on my deposit is poor compared to the market",
        ],
        "neutral": [
            "Interest rates are about average",
            "The deposit rate is okay, similar to others",
        ],
    },
    "branch_service": {
        "positive": [
            "Branch staff were friendly and efficient",
            "No queue at the branch and the service was quick",
            "The branch manager helped me patiently",
        ],
        "negative": [
            "Long queues at the branch and only one counter is open",
            "Branch staff were rude and unhelpful",
            "Wasted half a day at the branch for a simple request",
        ],
        "neutral": [
            "The branch visit was fine, a bit of a wait",
            "Branch service was average",
        ],
    },
    "upi_payments": {
        "positive": [
            "UPI payments are instant and never fail",
            "Digital payments are very convenient and secure",
            "Instant UPI transfers, works perfectly every time",
        ],
        "negative": [
            "UPI transactions fail often and the money gets stuck",
            "Refunds for failed UPI payments take days",
            "Payments are slow and the gateway keeps timing out",
        ],
        "neutral": [
            "UPI works fine most of the time",
            "Digital payments are okay, occasional delays",
        ],
    },
}


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _z(x):
    x = np.asarray(x, dtype=float)
    return (x - x.mean()) / (x.std() + 1e-9)


# --------------------------------------------------------------------------- #
# Customers + product usage
# --------------------------------------------------------------------------- #
def generate_customers_and_usage(n_customers: int = 3000, seed: int = 42):
    """Return (customers_clean, usage). Dirt is injected later by `_inject_dirt`."""
    rng = np.random.default_rng(seed)
    n = n_customers

    customer_id = np.array([f"CUST{i:05d}" for i in range(1, n + 1)])
    age = np.clip(rng.normal(38, 12, n), 21, 75).round().astype(int)
    gender = rng.choice(["Male", "Female", "Other"], n, p=[0.52, 0.47, 0.01])
    occupation = rng.choice(
        ["Salaried", "Self-Employed", "Business", "Student", "Retired", "Homemaker"],
        n,
        p=[0.45, 0.18, 0.12, 0.08, 0.10, 0.07],
    )
    # keep occupation consistent with age
    occupation = np.where((occupation == "Student") & (age > 28), "Salaried", occupation)
    occupation = np.where((occupation == "Retired") & (age < 50), "Self-Employed", occupation)
    city_tier = rng.choice(["Tier-1", "Tier-2", "Tier-3"], n, p=[0.40, 0.35, 0.25])
    tenure = np.clip(rng.gamma(2.0, 2.5, n), 0.2, 25).round(1)
    tenure = np.minimum(tenure, age - 18)

    occ_mult = {"Salaried": 1.0, "Self-Employed": 1.1, "Business": 1.5,
                "Student": 0.25, "Retired": 0.7, "Homemaker": 0.4}
    tier_mult = {"Tier-1": 1.25, "Tier-2": 1.0, "Tier-3": 0.8}
    income = (
        rng.lognormal(10.6, 0.5, n)
        * np.array([occ_mult[o] for o in occupation])
        * np.array([tier_mult[t] for t in city_tier])
    ).round(-2)  # monthly income (INR)

    zi = _z(np.log(income))
    za = _z(age)
    credit_score = np.clip(650 + 28 * zi + 4 * tenure + rng.normal(0, 45, n), 300, 900).round().astype(int)
    savings_rate = np.clip(0.16 + 0.05 * zi + 0.03 * za + rng.normal(0, 0.07, n), 0.0, 0.7).round(3)
    avg_balance = (income * rng.uniform(0.6, 5.0, n) * (1 + 3 * savings_rate)).round(-1)
    monthly_spend = (income * (1 - savings_rate) * rng.uniform(0.6, 1.0, n)).round(-1)

    digital = _sigmoid(1.2 - 0.06 * (age - 35) + rng.normal(0, 0.9, n))  # latent digital affinity
    mobile_app_user = (rng.random(n) < (0.10 + 0.85 * digital)).astype(int)
    login_count = mobile_app_user * rng.poisson(3 + 27 * digital)
    upi_txn = rng.poisson(2 + 40 * digital)
    card_txn = rng.poisson(1 + 12 * _sigmoid(zi))
    atm = rng.poisson(1 + 6 * (1 - digital))
    branch = rng.poisson(0.3 + 3 * (1 - digital))
    monthly_txn_count = upi_txn + card_txn + atm + rng.poisson(4, n)
    existing_emi_ratio = np.clip(rng.beta(1.5, 6, n), 0, 0.7).round(3)
    dependents = rng.poisson(np.clip((age - 22) / 20, 0.1, 2.0))
    has_salary_account = ((occupation == "Salaried") & (rng.random(n) < 0.8)).astype(int)

    # ---- hidden propensity model for product usage ------------------------- #
    zb = _z(np.log(avg_balance))
    zs = _z(savings_rate)
    zc = _z(credit_score)
    zd = _z(digital)
    z_card = _z(card_txn)
    z_login = _z(login_count)
    z_upi = _z(upi_txn)
    z_ten = _z(tenure)
    zsr = _z(monthly_spend / income)
    student = (occupation == "Student").astype(float)
    retired = (occupation == "Retired").astype(float)
    salaried = (occupation == "Salaried").astype(float)
    affluence = 0.6 * zi + 0.4 * zb + rng.normal(0, 0.4, n)  # shared latent factor

    logits = {
        "Fixed_Deposit": -0.9 + 0.9 * zb + 0.55 * za + 0.3 * zs + 0.6 * retired + 0.3 * affluence,
        "Credit_Card": -0.7 + 0.8 * zi + 0.9 * zc + 0.4 * z_card - 1.5 * student + 0.3 * affluence,
        "Personal_Loan": -1.1 - 0.7 * zb + 0.5 * zsr + 0.3 * zc + 0.5 * salaried - 0.8 * retired,
        "Home_Loan": -1.3 + 0.7 * zi + 0.5 * zc + 0.4 * z_ten + 0.3 * dependents - 0.07 * np.abs(age - 37),
        "Mutual_Fund_SIP": -1.0 + 0.7 * zi - 0.5 * za + 0.9 * zd + 0.8 * zs + 0.2 * affluence,
        "Digital_Banking_Plus": -0.6 + 1.2 * z_login + 0.7 * z_upi - 0.3 * za,
    }
    usage = pd.DataFrame({"customer_id": customer_id})
    for p in PRODUCTS:
        usage[p] = (rng.random(n) < _sigmoid(logits[p])).astype(int)

    customers = pd.DataFrame({
        "customer_id": customer_id,
        "age": age,
        "gender": gender,
        "occupation": occupation,
        "city_tier": city_tier,
        "tenure_years": tenure,
        "income": income,
        "credit_score": credit_score,
        "avg_balance": avg_balance,
        "monthly_spend": monthly_spend,
        "savings_rate": savings_rate,
        "monthly_txn_count": monthly_txn_count,
        "upi_txn_count": upi_txn,
        "card_txn_count": card_txn,
        "atm_withdrawals": atm,
        "branch_visits": branch,
        "login_count": login_count,
        "mobile_app_user": mobile_app_user,
        "existing_emi_ratio": existing_emi_ratio,
        "num_dependents": dependents,
        "has_salary_account": has_salary_account,
    })
    return customers, usage


def _inject_dirt(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Make the customer table realistically messy (to be repaired in preprocessing.py)."""
    rng = np.random.default_rng(seed + 7)
    df = df.copy()
    n = len(df)
    num_cols = [c for c in df.columns if c not in
                ("customer_id", "gender", "occupation", "city_tier")]
    df[num_cols] = df[num_cols].astype(float)  # so NaN / odd values can be assigned safely

    def pick(frac):
        return rng.choice(n, int(frac * n), replace=False)

    # 1) missing values
    for col, frac in [("income", 0.04), ("credit_score", 0.05), ("avg_balance", 0.03),
                      ("monthly_spend", 0.02), ("occupation", 0.02), ("gender", 0.02)]:
        df.loc[pick(frac), col] = np.nan

    # 2) inconsistent category labels
    i = pick(0.05)
    df.loc[i, "gender"] = df.loc[i, "gender"].map({"Male": "M", "Female": "female ", "Other": "O"})
    i = pick(0.04)
    df.loc[i, "city_tier"] = df.loc[i, "city_tier"].str.lower().str.replace("-", " ", regex=False)

    # 3) impossible / extreme values
    i = pick(0.005)
    df.loc[i, "age"] = rng.choice([-1, 0, 150, 200], len(i))
    i = pick(0.003)
    df.loc[i, "income"] = -df.loc[i, "income"].abs()
    i = pick(0.003)
    df.loc[i, "income"] = df.loc[i, "income"] * 60  # data-entry error (extra zeros)

    # 4) duplicated records
    dup = df.sample(frac=0.01, random_state=seed)
    return pd.concat([df, dup], ignore_index=True)


# --------------------------------------------------------------------------- #
# Feedback
# --------------------------------------------------------------------------- #
def _compose(topic: str, sent: str, rng) -> str:
    main = str(rng.choice(TEMPLATES[topic][sent]))
    if rng.random() < 0.18:  # mixed feedback -> makes the NLP task non-trivial
        other = str(rng.choice([t for t in TOPICS if t != topic]))
        opposite = {"positive": "negative", "negative": "positive", "neutral": "neutral"}[sent]
        extra = str(rng.choice(TEMPLATES[other][opposite]))
        main = f"{main}, but {extra[0].lower()}{extra[1:]}"
    opener = str(rng.choice(["", "", "Honestly, ", "Overall, ", "To be frank, "]))
    if opener:
        main = main[0].lower() + main[1:]
    return f"{opener}{main}."


def _rating(sent: str, rng) -> int:
    if rng.random() < 0.05:  # rating noise
        return int(rng.integers(1, 6))
    if sent == "positive":
        return int(rng.choice([4, 5], p=[0.4, 0.6]))
    if sent == "negative":
        return int(rng.choice([1, 2], p=[0.5, 0.5]))
    return int(rng.choice([2, 3, 4], p=[0.15, 0.70, 0.15]))


def generate_feedback(usage: pd.DataFrame, seed: int = 42, coverage: float = 0.65) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    dates = pd.date_range("2025-11-01", "2026-10-31", freq="D")
    channels = ["Mobile App", "Net Banking", "Branch", "Call Center", "Email"]
    own = usage.set_index("customer_id")
    rows, fid = [], 1
    for cid, r in own.iterrows():
        if rng.random() > coverage:
            continue
        for _ in range(1 if rng.random() < 0.75 else 2):
            w = np.ones(len(TOPICS))
            if r["Personal_Loan"] or r["Home_Loan"]:
                w[TOPICS.index("loan_service")] += 2
            if r["Credit_Card"]:
                w[TOPICS.index("credit_card")] += 2
            if r["Fixed_Deposit"]:
                w[TOPICS.index("interest_rates")] += 2
            if r["Digital_Banking_Plus"]:
                w[TOPICS.index("upi_payments")] += 1
                w[TOPICS.index("mobile_app")] += 1
            topic = TOPICS[rng.choice(len(TOPICS), p=w / w.sum())]
            sent = ["negative", "neutral", "positive"][rng.choice(3, p=TOPIC_SENT_PROBS[topic])]
            rows.append({
                "feedback_id": f"FB{fid:05d}",
                "customer_id": cid,
                "date": str(rng.choice(dates).astype("datetime64[D]")),
                "channel": str(rng.choice(channels)),
                "rating": _rating(sent, rng),
                "text": _compose(topic, sent, rng),
                "sentiment_label": sent,
            })
            fid += 1
    return pd.DataFrame(rows)


def generate_and_save(data_dir="data", n_customers: int = 3000, seed: int = 42):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    customers, usage = generate_customers_and_usage(n_customers, seed)
    feedback = generate_feedback(usage, seed)
    _inject_dirt(customers, seed).to_csv(data_dir / "customers_raw.csv", index=False)
    usage.to_csv(data_dir / "product_usage.csv", index=False)
    feedback.to_csv(data_dir / "feedback.csv", index=False)
    return data_dir


if __name__ == "__main__":
    path = generate_and_save()
    print(f"Synthetic data written to {path.resolve()}")
