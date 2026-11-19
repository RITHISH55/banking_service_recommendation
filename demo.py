"""
Command-line demonstration:  input data -> processing -> output.

Examples
  python demo.py                              # 3 random held-out customers
  python demo.py --customer CUST00042         # one existing customer
  python demo.py --new                        # brand-new customer (built-in example profile)
  python demo.py --new --profile '{"age": 52, "income": 40000, "occupation": "Retired"}' --owns Fixed_Deposit
  python demo.py --feedback "The app keeps crashing and support never picks up the phone"
Run `python main.py` once first (it trains and saves the models).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from data_generator import PRODUCTS
from inference import (EXAMPLE_PROFILE, analyse_feedback_text, label, load_bundle,
                       profile_to_features, score_existing_customer, score_features)

SHOW = ["age", "occupation", "city_tier", "income", "credit_score", "avg_balance",
        "savings_rate", "login_count", "upi_txn_count", "digital_txn_share"]


def bar(p, width=24):
    n = int(round(p * width))
    return "#" * n + "." * (width - n)


def show_result(title, feat_row, owned, probs, recs, extra=None):
    print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78)
    print("INPUT (profile):")
    for c in SHOW:
        v = feat_row[c]
        print(f"   {c:<18} {v:.2f}" if isinstance(v, (float, np.floating)) else f"   {c:<18} {v}")
    print("   currently holds   ", ", ".join(label(p) for p in owned) or "(no products yet)")
    print("\nPROCESSING: cleaning -> feature engineering -> per-service interest models -> hybrid ranking")
    print("\nPREDICTED INTEREST:")
    for p in probs.sort_values(ascending=False).index:
        tag = " (already holds)" if p in owned else ""
        print(f"   {label(p):<44} {bar(probs[p])} {probs[p]:>5.0%}{tag}")
    print("\nTOP RECOMMENDATIONS:")
    for _, r in recs.iterrows():
        print(f"   {r['rank']}. {r['product_name']}  -  {r['reason']}")
        if r["caution"]:
            print(f"      ! {r['caution']}")
    if extra:
        print("\n" + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--customer", help="existing customer id, e.g. CUST00042")
    ap.add_argument("--new", action="store_true", help="score a brand-new customer profile")
    ap.add_argument("--profile", help="JSON string overriding fields of the example profile")
    ap.add_argument("--owns", default="", help="comma-separated products the new customer already holds")
    ap.add_argument("--feedback", help="analyse the sentiment of a feedback text")
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()
    bundle = load_bundle()

    if args.feedback:
        r = analyse_feedback_text(bundle, args.feedback)
        print(f"\nFEEDBACK: {args.feedback}")
        print(f"Sentiment: {r['sentiment'].upper()}   Topic: {r['topic']}")
        print("Probabilities:", {k: round(float(v), 3) for k, v in r["probabilities"].items()})
        return

    if args.new:
        profile = dict(EXAMPLE_PROFILE)
        if args.profile:
            profile = json.loads(args.profile)  # fields left out are treated as missing and imputed
        owned = [p for p in args.owns.split(",") if p]
        bad = [p for p in owned if p not in PRODUCTS]
        if bad:
            raise SystemExit(f"Unknown products {bad}. Valid: {PRODUCTS}")
        feat = profile_to_features(profile)
        probs, recs = score_features(bundle, feat, owned, args.k)
        show_result("NEW CUSTOMER", feat.iloc[0], owned, probs, recs)
        return

    ids = [args.customer] if args.customer else list(
        np.random.default_rng().choice(bundle["test_ids"], 3, replace=False))
    for cid in ids:
        if cid not in bundle["customers"].index:
            raise SystemExit(f"Customer {cid} not found.")
        feat_row, owned, probs, recs = score_existing_customer(bundle, cid, args.k)
        fb = bundle["feedback_scored"]
        fb = fb[fb["customer_id"] == cid]
        extra = None
        if len(fb):
            last = fb.sort_values("date").iloc[-1]
            extra = (f"LATEST FEEDBACK ({last['date']}): \"{last['text']}\"\n"
                     f"   -> predicted sentiment: {last['pred_sentiment']}, topic: {last['topic']}")
        show_result(f"CUSTOMER {cid}", feat_row, owned, probs, recs, extra)


if __name__ == "__main__":
    main()
