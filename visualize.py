"""All charts used in the report. Every function saves a PNG and returns its path."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save files without opening windows
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_generator import PRODUCTS, PRODUCT_LABELS

sns.set_theme(style="whitegrid", font_scale=0.95)
# colour-blind-safe palette (Okabe-Ito)
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00",
     "sky": "#56B4E9", "pink": "#CC79A7", "grey": "#999999"}
SENT_COLORS = {"negative": C["red"], "neutral": C["grey"], "positive": C["green"]}


def _short(p):
    return p.replace("_", " ")


def _save(fig, out_dir, name):
    path = Path(out_dir) / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_adoption(own, out_dir):
    rate = own[PRODUCTS].mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([_short(p) for p in rate.index], rate.values, color=C["blue"])
    for i, v in enumerate(rate.values):
        ax.text(v + 0.5, i, f"{v:.1f}%", va="center")
    ax.set_xlabel("% of customers holding the service")
    ax.set_title("Current adoption of each banking service")
    ax.set_xlim(0, rate.max() * 1.15)
    return _save(fig, out_dir, "01_product_adoption.png")


def plot_model_comparison(results, out_dir):
    df = results.copy()
    df["product"] = df["product"].map(_short)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    palette = {"Baseline (prior)": C["grey"], "LogisticRegression": C["sky"],
               "RandomForest": C["green"], "GradientBoosting": C["blue"]}
    sns.barplot(data=df, x="product", y="test_roc_auc", hue="model", palette=palette, ax=ax)
    ax.axhline(0.5, color="black", lw=0.8, ls="--")
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_xlabel("")
    ax.set_title("Predicting customer interest: models vs baseline (higher is better)")
    ax.legend(loc="upper center", ncol=4, fontsize=8, frameon=False, bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out_dir, "02_model_comparison_auc.png")


def plot_feature_importance(importances, out_dir, top=7):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, p in zip(axes.ravel(), PRODUCTS):
        s = importances[p].head(top)[::-1]
        ax.barh(s.index, s.values, color=C["orange"])
        ax.set_title(_short(p))
        ax.set_xlabel("AUC drop when shuffled")
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("What drives customer interest? (permutation importance, test set)", y=1.0)
    return _save(fig, out_dir, "03_feature_importance.png")


def plot_reco_comparison(metrics, out_dir):
    cols = ["HitRate@1", "HitRate@2", "HitRate@3", "NDCG@3", "MRR"]
    long = metrics.melt(id_vars="method", value_vars=cols, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    pal = [C["grey"], C["sky"], C["orange"], C["green"], C["blue"]]
    sns.barplot(data=long, x="metric", y="value", hue="method", palette=pal, ax=ax)
    ax.set_ylim(0, 1)
    ax.set_xlabel("")
    ax.set_ylabel("Score (higher is better)")
    ax.set_title("Recommendation quality: leave-one-out on held-out customers")
    ax.legend(fontsize=8, ncol=2, frameon=False)
    return _save(fig, out_dir, "04_recommender_comparison.png")


def plot_alpha_sensitivity(sens, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sens["alpha"], sens["HitRate@1"], marker="o", color=C["blue"], label="HitRate@1")
    ax.plot(sens["alpha"], sens["HitRate@3"], marker="s", color=C["orange"], label="HitRate@3")
    ax.set_xlabel("alpha (weight of predictive model vs item-CF)")
    ax.set_ylabel("Hit rate")
    ax.set_title("Hybrid recommender: sensitivity to alpha")
    ax.legend(frameon=False)
    return _save(fig, out_dir, "05_hybrid_alpha_sensitivity.png")


def plot_sentiment_by_topic(summary, out_dir):
    s = summary.drop(index="other", errors="ignore").sort_values("pct_negative")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    left = np.zeros(len(s))
    for col, lab in [("pct_negative", "negative"), ("pct_neutral", "neutral"), ("pct_positive", "positive")]:
        ax.barh([t.replace("_", " ") for t in s.index], s[col], left=left, color=SENT_COLORS[lab], label=lab)
        left += s[col].to_numpy()
    ax.set_xlabel("% of feedback")
    ax.set_xlim(0, 100)
    ax.set_title("Customer feedback sentiment by service area")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    return _save(fig, out_dir, "06_sentiment_by_topic.png")


def plot_confusion(cm, labels, out_dir, title):
    fig, ax = plt.subplots(figsize=(4.8, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels,
                cbar=False, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    return _save(fig, out_dir, "07_sentiment_confusion.png")


def plot_segments(pca, labels, names, out_dir):
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for seg in sorted(labels.unique()):
        m = labels == seg
        ax.scatter(pca.loc[m, "pc1"], pca.loc[m, "pc2"], s=8, alpha=0.55, label=names[seg])
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Customer segments (K-Means, PCA projection)")
    ax.legend(fontsize=8, markerscale=2, frameon=False)
    return _save(fig, out_dir, "08_customer_segments.png")


def plot_segment_interest(probs, labels, names, out_dir):
    heat = probs.groupby(labels).mean() * 100
    heat.index = [names[i] for i in heat.index]
    heat.columns = [_short(c) for c in heat.columns]
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={"label": "avg predicted interest (%)"}, ax=ax)
    ax.set_title("Marketing view: average predicted interest by segment")
    ax.tick_params(axis="x", rotation=25)
    return _save(fig, out_dir, "09_segment_interest_heatmap.png")
