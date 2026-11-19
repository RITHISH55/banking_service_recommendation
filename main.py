"""
End-to-end pipeline:  data -> cleaning/features -> predictive models -> recommender
                      -> evaluation -> segmentation -> sentiment analysis -> outputs.

Run:  python main.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_generator import PRODUCTS, generate_and_save
from models import compute_importance, predict_interest, train_product_models
from preprocessing import (ALL_FEATURES, build_preprocessor, clean_customers,
                           engineer_features, feature_names)
from recommender import (AREA_TO_PRODUCTS, HybridRecommender, leave_one_out_split,
                         ranking_metrics)
from segmentation import segment_customers
from sentiment import (improvement_report, score_feedback, summarize_by_topic,
                       top_negative_terms, train_sentiment_models, LABELS)
import visualize as viz

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def banner(msg):
    print(f"\n{'=' * 78}\n{msg}\n{'=' * 78}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-customers", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--regenerate", action="store_true", help="re-create the synthetic dataset")
    ap.add_argument("--alpha", type=float, default=0.7, help="hybrid weight for the predictive model")
    args = ap.parse_args()
    seed = args.seed

    base = Path(__file__).parent
    data_dir, out_dir, fig_dir, model_dir = (base / "data", base / "outputs",
                                             base / "outputs" / "figures", base / "models")
    for d in (out_dir, fig_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ 1. data
    banner("1. DATA")
    if args.regenerate or not (data_dir / "customers_raw.csv").exists():
        generate_and_save(data_dir, args.n_customers, seed)
        print(f"Generated synthetic dataset in {data_dir}")
    raw = pd.read_csv(data_dir / "customers_raw.csv")
    usage = pd.read_csv(data_dir / "product_usage.csv")
    fb = pd.read_csv(data_dir / "feedback.csv")
    print(f"customers_raw: {raw.shape} | product_usage: {usage.shape} | feedback: {fb.shape}")

    # ---------------------------------------------------- 2. cleaning + features
    banner("2. DATA PROCESSING & FEATURE ENGINEERING")
    clean, report = clean_customers(raw)
    (out_dir / "data_quality_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not isinstance(v, dict)}, indent=2))
    print("missing values before cleaning:", report["missing_values_raw"])
    print("missing values after rule-based cleaning (imputed later in the pipeline):",
          report["missing_values_after_cleaning"])

    feat = engineer_features(clean).set_index("customer_id")
    own = usage.set_index("customer_id").loc[feat.index, PRODUCTS]
    # Convert the Arrow-backed pandas index to a NumPy array before splitting.
    # Recent scikit-learn versions otherwise index it with an integer array,
    # which pandas' Arrow array does not support.
    train_ids, test_ids = train_test_split(
        feat.index.to_numpy(), test_size=0.2, random_state=seed)
    pre = build_preprocessor().fit(feat.loc[train_ids, ALL_FEATURES])
    names = feature_names(pre)
    X_tr = pre.transform(feat.loc[train_ids, ALL_FEATURES])
    X_te = pre.transform(feat.loc[test_ids, ALL_FEATURES])
    X_all = pre.transform(feat[ALL_FEATURES])
    own_tr, own_te = own.loc[train_ids], own.loc[test_ids]
    print(f"train customers: {len(train_ids)} | test customers: {len(test_ids)} | "
          f"model features: {X_tr.shape[1]}")
    print("\nAdoption rates (%):\n", (own.mean() * 100).round(1).to_string())
    viz.plot_adoption(own, fig_dir)

    # ------------------------------------------------- 3. predictive modelling
    banner("3. PREDICTIVE MODELS - customer interest per service")
    results, best = train_product_models(X_tr, own_tr, X_te, own_te, seed)
    results.to_csv(out_dir / "model_metrics.csv", index=False)
    show = results.pivot(index="product", columns="model", values="test_roc_auc").round(3)
    print("Test ROC-AUC by product and model:\n", show.to_string())
    print("\nSelected model per product (by 5-fold CV on train):",
          {p: best[p][0] for p in PRODUCTS})
    sel = results[results["selected"]][["product", "model", "cv_roc_auc", "test_roc_auc",
                                       "test_pr_auc", "test_f1", "prevalence"]].round(3)
    print("\nSelected models - test metrics:\n", sel.to_string(index=False))
    viz.plot_model_comparison(results, fig_dir)
    viz.plot_feature_importance(compute_importance(best, X_te, own_te, names, seed), fig_dir)

    # -------------------------------------------- 4. recommender + evaluation
    banner("4. RECOMMENDER SYSTEM & EVALUATION (leave-one-out, held-out customers)")
    reco = HybridRecommender(PRODUCTS, alpha=args.alpha).fit(own_tr)
    eligible, masked, hidden = leave_one_out_split(own_te, PRODUCTS, seed)
    masked_df = pd.DataFrame(masked, columns=PRODUCTS, index=own_te.index[eligible])
    probs_masked = predict_interest(best, X_te[eligible], masked_df).to_numpy()
    rng = np.random.default_rng(seed)
    scores = {
        "Random": rng.random(masked.shape),
        "Popularity": reco.score_popularity(masked_df),
        "Item-CF": reco.score_itemcf(masked_df),
        "Model (predicted interest)": probs_masked,
        "Hybrid (model + CF)": reco.score_hybrid(probs_masked, masked_df),
    }
    metrics = pd.DataFrame([{"method": m, **ranking_metrics(s, masked, hidden)} for m, s in scores.items()])
    metrics.to_csv(out_dir / "recommendation_metrics.csv", index=False)
    print(f"Evaluated on {len(eligible)} test customers holding >= 2 services\n")
    print(metrics.round(3).to_string(index=False))
    viz.plot_reco_comparison(metrics, fig_dir)

    sens = pd.DataFrame([
        {"alpha": a, **{k: v for k, v in ranking_metrics(
            a * _mm(probs_masked) + (1 - a) * _mm(reco.score_itemcf(masked_df)), masked, hidden).items()
            if k in ("HitRate@1", "HitRate@3")}}
        for a in np.linspace(0, 1, 6)])
    viz.plot_alpha_sensitivity(sens, fig_dir)

    # --------------------------------------------------- 5. sentiment analysis
    banner("5. FEEDBACK SENTIMENT ANALYSIS")
    sent = train_sentiment_models(fb, seed)
    print(sent["results"].round(3).to_string(index=False))
    print(f"\nBest model: {sent['best_name']}\n{sent['report']}")
    scored = score_feedback(fb, sent["best_pipe"])
    scored.to_csv(out_dir / "feedback_scored.csv", index=False)
    summary = summarize_by_topic(scored)
    summary.to_csv(out_dir / "feedback_sentiment_summary.csv")
    terms = top_negative_terms(sent["logreg_pipe"])
    text = improvement_report(summary, terms)
    (out_dir / "feedback_improvements.txt").write_text(text)
    print("\n" + summary.to_string() + "\n\n" + text)
    viz.plot_sentiment_by_topic(summary, fig_dir)
    viz.plot_confusion(sent["confusion"], LABELS, fig_dir, sent["best_name"])
    sent["results"].to_csv(out_dir / "sentiment_model_metrics.csv", index=False)

    # customers with negative feedback in an area -> cross-sell caution flags
    neg = scored[(scored["pred_sentiment"] == "negative") & scored["topic"].isin(AREA_TO_PRODUCTS)]
    neg_areas = neg.groupby("customer_id")["topic"].apply(set).to_dict()

    # ----------------------------------- 6. predictions, segments, recommendations
    banner("6. PERSONALISED OUTPUTS")
    probs_all = predict_interest(best, X_all, own)
    pi = probs_all.round(4).copy()
    pi.insert(0, "split", np.where(pi.index.isin(set(test_ids)), "test", "train"))
    pi.to_csv(out_dir / "predicted_interest.csv")
    recs = reco.recommend_batch(own, probs_all, k=3, neg_areas=neg_areas)
    recs.to_csv(out_dir / "recommendations.csv", index=False)
    print("Sample recommendations:\n", recs.head(9)[["customer_id", "rank", "product_name",
                                                    "interest_prob", "reason"]].to_string(index=False))
    print(f"\nCross-sell caution flags raised: {(recs['caution'] != '').sum()} of {len(recs)} recommendations")

    seg = segment_customers(feat, seed=seed)
    seg["profile"].round(2).assign(segment_name=pd.Series(seg["names"])).to_csv(out_dir / "segment_profile.csv")
    seg["labels"].to_csv(out_dir / "customer_segments.csv")
    print(f"\nSegments: k={seg['best_k']} (silhouette {max(seg['silhouette'].values()):.3f})")
    print(seg["profile"].round(2).assign(name=pd.Series(seg["names"])).to_string())
    viz.plot_segments(seg["pca"], seg["labels"], seg["names"], fig_dir)
    viz.plot_segment_interest(probs_all, seg["labels"], seg["names"], fig_dir)

    # ------------------------------------------------------------- 7. persist
    joblib.dump({
        "preprocessor": pre, "feature_names": names, "models": best, "recommender": reco,
        "sentiment_pipe": sent["best_pipe"], "customers": feat, "ownership": own,
        "neg_areas": neg_areas, "feedback_scored": scored, "test_ids": list(test_ids),
        "segments": seg["labels"], "segment_names": seg["names"],
    }, model_dir / "bundle.joblib")
    banner("DONE")
    print(f"Tables : {out_dir}\nFigures: {fig_dir}\nModel  : {model_dir / 'bundle.joblib'}")
    print("\nTry the demo:  python demo.py")


def _mm(a):
    lo, hi = a.min(1, keepdims=True), a.max(1, keepdims=True)
    return (a - lo) / np.where(hi - lo == 0, 1, hi - lo)


if __name__ == "__main__":
    main()
