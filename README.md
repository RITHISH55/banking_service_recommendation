# Personalized Banking Service Recommendation

A recommender system that suggests banking services (savings products, credit cards, loans,
digital-banking features) to each customer, predicts how interested they are in each service,
and mines customer feedback with sentiment analysis to find service improvements.

> Syllabus mapping — **Unit 4:** predictive modelling, unstructured data (text), feature engineering.
> **Unit 5:** customer insights, marketing analysis (segmentation), sentiment analysis, recommender systems.

---

## 1. Problem statement
A bank wants to cross-sell the *right* service to the *right* customer. Blanket campaigns waste
money and annoy customers. Using profile data, account activity, current product holdings and
free-text feedback, we must (a) estimate each customer's interest in every service, (b) turn
that into a ranked, explainable recommendation list, (c) evaluate the recommendations offline, and
(d) read customer feedback to see where the bank's own service needs fixing first.

## 2. Objectives
1. Build a clean, documented dataset (profile + activity + product usage + feedback).
2. Clean it and engineer behavioural / financial-health features.
3. Train predictive models of customer interest per service and compare them with a baseline.
4. Implement a recommender (popularity, item-CF, model-based, hybrid) and evaluate it.
5. Classify feedback sentiment, tag service topics, and produce improvement priorities.
6. Segment customers for marketing analysis.
7. Provide a working demonstration (CLI + optional web UI).

## 3. Dataset (documented synthetic data)
No public dataset links profiles, product holdings **and** feedback for the same bank customers,
so `data_generator.py` creates one (seeded, reproducible: `seed=42`, 3,000 customers).

| File | Rows | Description |
|---|---|---|
| `data/customers_raw.csv` | 3,030 | 21 columns: age, gender, occupation, city tier, tenure, monthly income (INR), credit score, balance, spend, savings rate, UPI/card/ATM/branch counts, app logins, EMI ratio, dependents, … |
| `data/product_usage.csv` | 3,000 | 0/1 holding of 6 services: Fixed Deposit, Credit Card, Personal Loan, Home Loan, Mutual Fund SIP, Digital Banking Plus |
| `data/feedback.csv` | ~2,480 | free-text feedback, rating (1-5), channel, date, sentiment label |

**How it is generated.** Product holding is drawn from a hidden logistic model of customer
features plus a shared "affluence" factor (so products correlate, as in real banks). Feedback
text is composed from topic × sentiment templates (8 topics, ~18 % "mixed" sentences, noisy ratings);
topics like fees and customer support are made genuinely more negative so there is something to find.

**Deliberate data problems** (fixed in preprocessing): ~2-5 % missing values in 6 columns, inconsistent
labels (`M`, `female `, `tier 1`), impossible ages (-1, 0, 150, 200), negative incomes, ×60 income
typos, and 1 % duplicated rows.

**Limitation:** because the data is synthetic, absolute scores (especially sentiment) are more optimistic
than on real bank data. Replace the three CSVs with real data of the same schema and the code runs unchanged.
Interest is modelled with *implicit feedback* (current holding = positive label).

## 4. Data processing (`preprocessing.py`)
* **Cleaning (rule-based, no leakage):** duplicates dropped; gender / city-tier labels standardised; impossible
  values → missing; business-rule outlier cap (income > 2 M/month = entry error); missing categories → `Unknown`.
* **Imputation / scaling (fitted on the training split only):** median imputation with missing-indicator flags,
  standard scaling, one-hot encoding.
* **Engineered features:** `income_log`, `balance_log`, `spend_to_income`, `balance_to_income_months`,
  `digital_txn_share`, `branch_dependence`, `high_emi_flag`, `is_new_customer`, `age_band`, `credit_band`.

## 5. Algorithms
| Task | Methods | Baseline |
|---|---|---|
| Interest prediction (6 binary targets) | Logistic Regression, Random Forest, Gradient Boosting (best chosen by 5-fold CV) | Class-prior dummy |
| Recommendation | Item-based CF (cosine co-holding), model-based ranking, **Hybrid** = α·model + (1-α)·item-CF (α = 0.7) | Random, Popularity |
| Sentiment (neg / neutral / pos) | TF-IDF (1-2 grams) + Logistic Regression, TF-IDF + Naive Bayes | Lexicon with negation handling |
| Topic tagging | Transparent keyword/regex rules | – |
| Segmentation | K-Means (k by silhouette) + PCA for plotting | – |

For target service *p*, features = customer features + holding flags of the **other** services (never *p* itself).

## 6. Installation
Requires Python 3.9+.
```bash
cd banking_reco
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
```
(`streamlit` is only needed for the optional web UI.)

## 7. Execution
```bash
python main.py              # generates data (first run), trains everything, writes outputs/ and models/
python demo.py              # demo on 3 random held-out customers
python demo.py --customer CUST00004
python demo.py --new        # a brand-new customer (built-in example)
python demo.py --new --profile '{"age": 52, "income": 40000, "occupation": "Retired"}' --owns Fixed_Deposit
python demo.py --feedback "The app keeps crashing and support never picks up the phone"
streamlit run app.py        # optional web UI
```
On Windows PowerShell wrap the JSON like this: `--profile "{\"age\": 52, \"income\": 40000}"`.
Options for `main.py`: `--regenerate`, `--n-customers 5000`, `--seed 7`, `--alpha 0.5`.
Fields omitted from `--profile` are treated as missing and imputed — a demonstration of the missing-data handling.

## 8. Outputs
| File (in `outputs/`) | Content |
|---|---|
| `recommendations.csv` | top-3 services per customer with explanation and cross-sell caution flag |
| `predicted_interest.csv` | predicted interest probability for every customer × service |
| `feedback_sentiment_summary.csv`, `feedback_improvements.txt`, `feedback_scored.csv` | sentiment by topic and improvement priorities |
| `model_metrics.csv`, `recommendation_metrics.csv`, `sentiment_model_metrics.csv` | evaluation tables |
| `segment_profile.csv`, `customer_segments.csv` | segmentation |
| `data_quality_report.json` | what was cleaned |
| `figures/*.png` | 9 charts (adoption, model comparison, feature importance, recommender comparison, α sensitivity, sentiment by topic, confusion matrix, segments, segment-interest heatmap) |

## 9. Sample results (seed 42)
**Predictive models** — test ROC-AUC (baseline = 0.50): Digital Banking Plus 0.85, Credit Card 0.84,
Fixed Deposit 0.82, Home Loan 0.79, Mutual Fund SIP 0.78, Personal Loan 0.73.

**Recommender** — leave-one-out on 383 held-out customers holding ≥ 2 services:

| Method | HitRate@1 | HitRate@3 | NDCG@3 | MRR |
|---|---|---|---|---|
| Random | 0.300 | 0.755 | 0.561 | 0.552 |
| Popularity | 0.272 | 0.812 | 0.585 | 0.551 |
| Item-CF | 0.426 | 0.859 | 0.676 | 0.647 |
| Model (predicted interest) | 0.590 | 0.927 | 0.789 | 0.759 |
| Hybrid (model + CF) | 0.590 | 0.940 | 0.794 | 0.758 |

Personalised ranking roughly doubles HitRate@1 over random/popularity. The hybrid is essentially tied with the
model alone (slightly better at @3). With only 6 services, HitRate@3 is easy even for random, so focus on @1, NDCG and MRR.

**Sentiment** — macro-F1: Lexicon 0.84 → Naive Bayes 0.96 → TF-IDF + LogReg 0.985 (accuracy 98 %; inflated by template-based synthetic text).
Highest negative share: **fees & charges 62 %**, **customer support 53 %**, **loan service 45 %**, interest rates 41 %.
Lowest: mobile app 28 %. Suggested actions are in `outputs/feedback_improvements.txt`.

**Sample recommendation (CUST00004, holds Home Loan):** 1. Credit Card (69 % interest; often held with Home Loan),
2. Fixed Deposit (45 %), 3. Personal Loan (30 %).

## 10. Project structure
```
banking_reco/
├── data_generator.py   synthetic dataset
├── preprocessing.py    cleaning, feature engineering, sklearn preprocessor
├── models.py           per-service predictive models + importance
├── recommender.py      popularity / item-CF / hybrid + leave-one-out evaluation
├── sentiment.py        sentiment models, topic tagging, improvement report
├── segmentation.py     K-Means customer segments
├── visualize.py        all charts
├── main.py             runs the whole pipeline
├── inference.py        shared scoring helpers
├── demo.py             command-line demonstration
├── app.py              optional Streamlit UI
└── requirements.txt
```

## 11. Notes and limitations
* Predicted interest for *training* customers in `predicted_interest.csv` is in-sample (see its `split` column); metrics are computed on the held-out test split only.
* Holding a product is a proxy for interest, not proof that a non-holder wants it. A real deployment should validate with campaign response data (A/B tests).
* Cross-sell caution flags: if a customer left negative feedback about an area (e.g. loan service), the related product is flagged "resolve the issue first".
* In production, use a time-based train/test split and add fairness checks (e.g. credit products across demographic groups).
