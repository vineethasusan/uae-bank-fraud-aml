"""End-to-end run: data -> card fraud model -> AML scenarios -> alert triage -> reports.

    python -m fincrime.pipeline
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import pandas as pd

from . import aml, fraud
from . import data as datagen

OUT = Path("outputs")


def ensure_data():
    if not Path("data/customers.csv").exists():
        Path("data").mkdir(exist_ok=True)
        cust, cards, acct = datagen.generate()
        cust.to_csv("data/customers.csv", index=False)
        cards.to_csv("data/card_transactions.csv.gz", index=False)
        acct.to_csv("data/account_transactions.csv.gz", index=False)


def main():
    t0 = time.time()
    ensure_data()
    OUT.mkdir(exist_ok=True)
    cust = pd.read_csv("data/customers.csv")
    cards = pd.read_csv("data/card_transactions.csv.gz", dtype={"_fraud_type": "object"})
    acct = pd.read_csv("data/account_transactions.csv.gz", parse_dates=["ts"])

    print("1/3 Card fraud: features and model…")
    feat = fraud.build_features(cards, cust)
    res = fraud.train_and_evaluate(feat)
    test = res["test"]
    ref = feat[feat["ts"] <= fraud.VALID_END].sample(20000, random_state=0)
    queue = test[test["alert"] == 1].sort_values("score", ascending=False).head(300).copy()
    queue["reasons"] = [" · ".join(fraud.reason_codes(res["model"], r, ref)) for _, r in queue.iterrows()]
    queue[["txn_id", "ts", "customer_id", "mcc", "channel", "country", "amount_aed", "score", "reasons", "is_fraud"]] \
        .to_csv(OUT / "fraud_alert_queue.csv", index=False)
    joblib.dump({"model": res["model"], "threshold": res["threshold"], "features": fraud.FEATURES}, OUT / "fraud_model.joblib")

    print("2/3 AML: scenarios…")
    alerts = aml.run_scenarios(acct, cust)
    cov, per = aml.scenario_coverage(alerts, cust)
    print("3/3 AML: triage model…")
    af = aml.build_alert_features(alerts, acct, cust)
    tm, ranked, tmet = aml.train_triage(af)
    ranked[["customer_id", "scenario_name", "alert_date", "amount_aed", "priority", "linked_alerted_customers",
            "foreign_out_share", "anomaly_score", "_true_suspicious"]].to_csv(OUT / "aml_alert_queue.csv", index=False)

    m = res["metrics"]
    metrics = {"fraud": {k: v for k, v in m.items()}, "aml_scenario_coverage": cov.to_dict(orient="records"),
               "aml_scenario_precision": per.reset_index().to_dict(orient="records"), "aml_triage": tmet,
               "runtime_s": round(time.time() - t0)}
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, default=float))
    write_report(metrics)
    print(f"Done in {metrics['runtime_s']}s -> outputs/report.md")


def write_report(m):
    f = m["fraud"]
    mo, ru, na = f["model"], f["rules_baseline"], f["no_action"]
    L = ["# Financial crime analytics: results", "", "## Card fraud (test period: May–June 2026)", "",
         f"{f['test_transactions']:,} transactions, fraud rate {f['fraud_rate']:.3%}. "
         f"ROC-AUC {f['roc_auc']}, PR-AUC {f['pr_auc']}.", "",
         "| Approach | Alerts | Fraud caught | Precision | Fraud loss | Total cost (loss + reviews) |", "|---|---|---|---|---|---|"]
    for name, r in [("No monitoring", na), ("Legacy rules", ru), ("ML model (cost-optimised threshold)", mo)]:
        L.append(f"| {name} | {r['alerts']:,} | {r['fraud_caught']}/{r['fraud_total']} | {r['precision']:.0%} | "
                 f"AED {r['fraud_value_missed_aed']:,} | AED {r['total_cost_aed']:,} |")
    L += ["", "Detection rate by fraud pattern: " + ", ".join(f"{k.replace('_', ' ')} {v:.0%}"
                                                             for k, v in f["detection_rate_by_fraud_type"].items()), "",
          "## AML monitoring", "", "**Scenario coverage of known typologies**", "",
          "| Typology | Customers | Detected | Coverage |", "|---|---|---|---|"]
    L += [f"| {r['typology'].replace('_', ' ')} | {r['customers']} | {r['detected_by_any_scenario']} | {r['coverage']:.0%} |"
          for r in m["aml_scenario_coverage"]]
    L += ["", "**Scenario precision** (share of alerts that are truly suspicious)", "", "| Scenario | Alerts | Precision |",
          "|---|---|---|"]
    L += [f"| {r['scenario_name']} | {r['alerts']:,} | {r['precision']:.0%} |" for r in m["aml_scenario_precision"]]
    t = m["aml_triage"]
    L += ["", "**Alert triage (Apr–Jun 2026 alerts, model trained on Jan–Mar decisions)**", "",
          f"- {t['alerts_test']:,} alerts, of which {t['truly_suspicious']} truly suspicious "
          f"({t['false_positive_rate_of_rules']:.0%} false positives from rules alone)",
          f"- Top 20% of the ranked queue contains **{t['suspicious_found_in_top_20pct']:.0%}** of suspicious cases",
          f"- 90% of suspicious cases are found after reviewing **{t['alerts_to_review_for_90pct_recall']}** alerts: "
          f"a **{t['workload_reduction_at_90pct_recall']:.0%} workload reduction**"]
    (OUT / "report.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
