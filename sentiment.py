"""
Sentiment analysis of customer feedback + topic tagging + improvement summary.

Models compared (3-class: negative / neutral / positive)
  Lexicon (baseline)      - hand-made word lists with simple negation handling
  TF-IDF + NaiveBayes     - uni+bi-gram TF-IDF, Multinomial NB
  TF-IDF + LogReg         - uni+bi-gram TF-IDF, class-balanced logistic regression
Topics are tagged with transparent keyword rules (no ground truth used).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

LABELS = ["negative", "neutral", "positive"]

POS_WORDS = {
    "good", "great", "smooth", "easy", "fast", "quick", "quickly", "helpful", "friendly", "polite",
    "responsive", "love", "happy", "attractive", "convenient", "reliable", "instant", "generous",
    "clear", "transparent", "efficient", "resolved", "competitive", "excellent", "perfectly",
    "secure", "patiently", "simple", "solved", "minimal", "reasonable",
}
NEG_WORDS = {
    "crashing", "crashes", "slow", "fails", "fail", "painfully", "glitches", "delayed", "rude",
    "unhelpful", "hidden", "unexpected", "unfair", "high", "long", "unresolved", "blocked",
    "expiring", "weeks", "stuck", "lower", "poor", "freezes", "errors", "wasted", "confusing",
    "increasing", "timing", "different", "paperwork", "penalty",
}
NEGATORS = {"not", "no", "never", "without", "n't"}


def lexicon_predict(texts) -> list[str]:
    preds = []
    for t in texts:
        toks = re.findall(r"[a-z']+", t.lower())
        score = 0
        for i, w in enumerate(toks):
            val = 1 if w in POS_WORDS else -1 if w in NEG_WORDS else 0
            if val and any(x in NEGATORS for x in toks[max(0, i - 2):i]):
                val = -val
            score += val
        preds.append("positive" if score > 0 else "negative" if score < 0 else "neutral")
    return preds


def _pipe(clf) -> Pipeline:
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                     ("clf", clf)])


def train_sentiment_models(fb: pd.DataFrame, seed: int = 42):
    X_tr, X_te, y_tr, y_te = train_test_split(
        fb["text"], fb["sentiment_label"], test_size=0.2, stratify=fb["sentiment_label"],
        random_state=seed)
    pipes = {
        "TF-IDF + NaiveBayes": _pipe(MultinomialNB(alpha=0.5)),
        "TF-IDF + LogReg": _pipe(LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced")),
    }
    rows = []
    pred = lexicon_predict(X_te)
    rows.append({"model": "Lexicon (baseline)", "accuracy": accuracy_score(y_te, pred),
                 "macro_f1": f1_score(y_te, pred, average="macro")})
    preds = {"Lexicon (baseline)": pred}
    for name, pipe in pipes.items():
        pipe.fit(X_tr, y_tr)
        preds[name] = pipe.predict(X_te)
        rows.append({"model": name, "accuracy": accuracy_score(y_te, preds[name]),
                     "macro_f1": f1_score(y_te, preds[name], average="macro")})
    results = pd.DataFrame(rows)
    best_name = results[results["model"] != "Lexicon (baseline)"].sort_values(
        "macro_f1", ascending=False).iloc[0]["model"]
    cm = confusion_matrix(y_te, preds[best_name], labels=LABELS)
    report = classification_report(y_te, preds[best_name], labels=LABELS, digits=3)
    return {"results": results, "best_name": best_name, "best_pipe": pipes[best_name],
            "logreg_pipe": pipes["TF-IDF + LogReg"], "confusion": cm, "report": report}


# ---- topic tagging ------------------------------------------------------- #
TOPIC_PATTERNS = {
    "loan_service": r"\bloans?\b|\bdisburs|\bpaperwork|\bdocument|\bapproval|\bemi\b",
    "credit_card": r"\bcredit card|\bcard\b|\breward|\bcashback|\bannual fee",
    "customer_support": r"\bsupport|\bcustomer care|\bagents?\b|\bon hold|\bcall",
    "fees_charges": r"\bfees?\b|\bcharges?\b|\bpenalty",
    "interest_rates": r"\binterest|\brates?\b|\bfd\b|\bfixed deposit|\bdeposit",
    "upi_payments": r"\bupi\b|\bdigital payments?|\brefunds?\b|\bpayments?\b|\bgateway",
    "mobile_app": r"\bapps?\b|\bcrash|\bglitch|\blogin\b|\bfreez",
    "branch_service": r"\bbranch|\bqueues?\b|\bcounter|\bstaff",
}
_COMPILED = {t: re.compile(p) for t, p in TOPIC_PATTERNS.items()}

ACTIONS = {
    "loan_service": "Digitise loan onboarding (e-KYC, document upload), publish an SLA and send status updates.",
    "credit_card": "Review annual-fee waivers and reward-point expiry rules; add proactive block/unblock alerts.",
    "customer_support": "Cut call wait times (callback option, chatbot triage) and standardise agent answers.",
    "fees_charges": "Make the fee schedule transparent, add pre-deduction alerts and review minimum-balance penalties.",
    "interest_rates": "Benchmark deposit and loan rates against competitors; offer relationship-based rate tiers.",
    "upi_payments": "Improve failed-transaction auto-reversal and communicate refund timelines in-app.",
    "mobile_app": "Prioritise crash/performance fixes in the app, improve login speed and release testing.",
    "branch_service": "Add token/queue management and extend counter staffing at peak hours.",
}


def tag_topic(text: str) -> str:
    counts = {t: len(p.findall(text.lower())) for t, p in _COMPILED.items()}
    top = max(counts, key=counts.get)
    return top if counts[top] > 0 else "other"


def score_feedback(fb: pd.DataFrame, pipe) -> pd.DataFrame:
    out = fb.copy()
    out["topic"] = out["text"].map(tag_topic)
    out["pred_sentiment"] = pipe.predict(out["text"])
    proba = pipe.predict_proba(out["text"])
    classes = list(pipe.classes_)
    out["p_negative"] = proba[:, classes.index("negative")]
    out["sentiment_score"] = proba[:, classes.index("positive")] - proba[:, classes.index("negative")]
    return out


def summarize_by_topic(scored: pd.DataFrame) -> pd.DataFrame:
    g = scored.groupby("topic")
    summary = pd.DataFrame({
        "n_feedback": g.size(),
        "avg_rating": g["rating"].mean().round(2),
        "pct_negative": g["pred_sentiment"].apply(lambda s: (s == "negative").mean() * 100).round(1),
        "pct_neutral": g["pred_sentiment"].apply(lambda s: (s == "neutral").mean() * 100).round(1),
        "pct_positive": g["pred_sentiment"].apply(lambda s: (s == "positive").mean() * 100).round(1),
        "avg_sentiment_score": g["sentiment_score"].mean().round(3),
    })
    summary["negative_count"] = (summary["n_feedback"] * summary["pct_negative"] / 100).round().astype(int)
    return summary.sort_values("negative_count", ascending=False)


def top_negative_terms(logreg_pipe, n: int = 15) -> pd.Series:
    vec, clf = logreg_pipe.named_steps["tfidf"], logreg_pipe.named_steps["clf"]
    coef = clf.coef_[list(clf.classes_).index("negative")]
    names = np.array(vec.get_feature_names_out())
    idx = np.argsort(coef)[::-1][:n]
    return pd.Series(coef[idx], index=names[idx])


def improvement_report(summary: pd.DataFrame, terms: pd.Series) -> str:
    lines = ["SERVICE IMPROVEMENT PRIORITIES (from predicted feedback sentiment)", "=" * 66]
    for rank, (topic, r) in enumerate(summary.iterrows(), start=1):
        if topic == "other":
            continue
        flag = "HIGH" if r["pct_negative"] >= 45 else "MEDIUM" if r["pct_negative"] >= 35 else "LOW"
        lines.append(f"{rank}. {topic.replace('_', ' ').title():<18} negative {r['pct_negative']:>5.1f}%  "
                     f"({int(r['negative_count'])} of {int(r['n_feedback'])})  avg rating {r['avg_rating']}  [{flag}]")
        lines.append(f"     Action: {ACTIONS.get(topic, 'Investigate root causes.')}")
    lines += ["", "Strongest negative-sentiment terms: " + ", ".join(terms.index[:12])]
    return "\n".join(lines)
