"""AML transaction monitoring: scenario rules -> alerts -> ML triage of the investigator queue.

Banks run typology scenarios (rules) because regulators expect explainable coverage, but
rules produce many false positives. Here rules stay as the safety net, and a triage model,
trained on past investigator decisions, ranks alerts so analysts work the riskiest first.
Features include network signals (counterparties shared with other alerted customers) and
an unsupervised anomaly score.

Scenarios
  S1 structuring       >= 3 cash deposits at 80-100% of the reporting threshold within 14 days
  S2 pass-through      inflow >= AED 100k with >= 90% leaving the account within 72 hours
  S3 fan-in mule       >= 8 distinct senders within 7 days, total > 2x salary, then outflow abroad
  S4 high-risk corridor>= 3 transfers to high-risk jurisdictions within 30 days
  S5 income mismatch   monthly inflows > 5x declared salary
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from .data import HIGH_RISK, REPORTING_THRESHOLD

SCENARIOS = {"S1": "Structuring", "S2": "Rapid pass-through", "S3": "Fan-in / money mule",
             "S4": "High-risk corridor", "S5": "Income mismatch"}
TYPOLOGY_FOR = {"S1": "structuring", "S2": "rapid_pass_through", "S3": "money_mule", "S4": "high_risk_corridor"}


def run_scenarios(acct: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    acct = acct.copy()
    acct["ts"] = pd.to_datetime(acct["ts"])
    salary = cust.set_index("customer_id")["monthly_salary_aed"]
    alerts = []
    for cid, g in acct.groupby("customer_id"):
        g = g.sort_values("ts")
        ts = g["ts"].to_numpy()
        # S1 structuring
        near = g[(g["type"] == "cash_deposit") & (g["amount_aed"].between(0.8 * REPORTING_THRESHOLD, REPORTING_THRESHOLD - 1))]
        for i in range(len(near)):
            w = near[(near["ts"] >= near["ts"].iloc[i]) & (near["ts"] <= near["ts"].iloc[i] + pd.Timedelta(days=14))]
            if len(w) >= 3:
                alerts.append((cid, "S1", w["ts"].iloc[-1], float(w["amount_aed"].sum())))
                break
        # S2 pass-through
        ins = g[(g["direction"] == "credit") & (g["amount_aed"] >= 100_000) & (g["type"] != "salary_in")]
        outs = g[g["direction"] == "debit"]
        for t, a in zip(ins["ts"], ins["amount_aed"]):
            o = outs[(outs["ts"] > t) & (outs["ts"] <= t + pd.Timedelta(hours=72))]["amount_aed"].sum()
            if o >= 0.9 * a:
                alerts.append((cid, "S2", t + pd.Timedelta(hours=72), float(a)))
                break
        # S3 fan-in
        tin = g[g["type"] == "transfer_in"]
        for t in tin["ts"]:
            w = tin[(tin["ts"] >= t) & (tin["ts"] <= t + pd.Timedelta(days=7))]
            if w["counterparty"].nunique() >= 8 and w["amount_aed"].sum() > 2 * salary[cid]:
                end = w["ts"].max()
                abroad = g[(g["direction"] == "debit") & (g["ts"] > t) & (g["ts"] <= end + pd.Timedelta(days=7))
                           & (g["counterparty_country"] != "AE")]
                if len(abroad):
                    alerts.append((cid, "S3", end, float(w["amount_aed"].sum())))
                    break
        # S4 high-risk corridor
        hr = g[(g["direction"] == "debit") & g["counterparty_country"].isin(HIGH_RISK)]
        for t in hr["ts"]:
            w = hr[(hr["ts"] >= t) & (hr["ts"] <= t + pd.Timedelta(days=30))]
            if len(w) >= 3:
                alerts.append((cid, "S4", w["ts"].iloc[2], float(w["amount_aed"].sum())))
                break
        # S5 income mismatch (by month)
        cr = g[g["direction"] == "credit"]
        monthly = cr.groupby(cr["ts"].dt.to_period("M"))["amount_aed"].sum()
        for per, total in monthly.items():
            if total > 5 * salary[cid]:
                alerts.append((cid, "S5", per.to_timestamp(how="end").normalize(), float(total)))
                break
        _ = ts
    a = pd.DataFrame(alerts, columns=["customer_id", "scenario", "alert_date", "amount_aed"])
    a["scenario_name"] = a["scenario"].map(SCENARIOS)
    return a.sort_values("alert_date").reset_index(drop=True)


def network_features(acct: pd.DataFrame, alerted: set) -> pd.DataFrame:
    """Customer <-> counterparty graph. Mules and launderers share counterparties with other suspects."""
    G = nx.Graph()
    ext = acct[acct["counterparty"].str.startswith("EXT")]
    G.add_edges_from(zip(ext["customer_id"], ext["counterparty"]))
    rows = []
    for c in acct["customer_id"].unique():
        if c not in G:
            rows.append((c, 0, 0))
            continue
        cps = list(G.neighbors(c))
        shared = {n for cp in cps for n in G.neighbors(cp) if n != c and n.startswith("CU")}
        rows.append((c, len(cps), len(shared & alerted)))
    return pd.DataFrame(rows, columns=["customer_id", "n_counterparties", "linked_alerted_customers"]).set_index("customer_id")


def customer_profile(acct: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    a = acct.copy()
    a["ts"] = pd.to_datetime(a["ts"])
    c = cust.set_index("customer_id")
    g = a.groupby("customer_id")
    prof = pd.DataFrame(index=c.index)
    prof["inflow_total"] = a[a["direction"] == "credit"].groupby("customer_id")["amount_aed"].sum()
    prof["outflow_total"] = a[a["direction"] == "debit"].groupby("customer_id")["amount_aed"].sum()
    prof["cash_total"] = a[a["type"] == "cash_deposit"].groupby("customer_id")["amount_aed"].sum()
    prof["foreign_out_share"] = a[a["direction"] == "debit"].assign(f=lambda d: d["counterparty_country"] != "AE") \
        .groupby("customer_id")["f"].mean()
    prof["distinct_senders"] = a[a["type"] == "transfer_in"].groupby("customer_id")["counterparty"].nunique()
    prof["txn_count"] = g.size()
    prof = prof.fillna(0)
    prof["inflow_to_salary"] = prof["inflow_total"] / (6 * c["monthly_salary_aed"])
    prof["cash_to_salary"] = prof["cash_total"] / (6 * c["monthly_salary_aed"])
    prof["anomaly_score"] = -IsolationForest(n_estimators=200, random_state=0).fit(
        np.log1p(prof)).score_samples(np.log1p(prof))
    return prof


TRIAGE_FEATURES = ["scenario_code", "amount_to_salary", "n_scenarios_hit", "account_age_months", "occupation_risk_code",
                   "segment_code", "inflow_to_salary", "cash_to_salary", "foreign_out_share", "distinct_senders",
                   "n_counterparties", "linked_alerted_customers", "anomaly_score"]


def build_alert_features(alerts: pd.DataFrame, acct: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    c = cust.set_index("customer_id")
    prof = customer_profile(acct, cust)
    net = network_features(acct, set(alerts["customer_id"]))
    f = alerts.join(prof, on="customer_id").join(net, on="customer_id")
    f["scenario_code"] = f["scenario"].str[1].astype(int)
    f["amount_to_salary"] = f["amount_aed"] / f["customer_id"].map(c["monthly_salary_aed"])
    f["n_scenarios_hit"] = f.groupby("customer_id")["scenario"].transform("nunique")
    f["account_age_months"] = f["customer_id"].map(c["account_age_months"])
    f["occupation_risk_code"] = f["customer_id"].map(c["occupation_risk"].map({"low": 0, "medium": 1, "high": 2}))
    f["segment_code"] = f["customer_id"].map(c["segment"].map({"Mass": 0, "Affluent": 1, "Private": 2}))
    # Historical investigator decision (label): suspicious if the customer really follows a typology.
    # Investigators are not perfect: ~5% of true cases were closed, ~2% of false ones escalated.
    truth = f["customer_id"].map(c["_aml_typology"]).notna().astype(int)
    rng = np.random.default_rng(1)
    flip = rng.random(len(f))
    f["sar_filed"] = np.where(truth == 1, (flip > 0.05).astype(int), (flip < 0.02).astype(int))
    f["_true_suspicious"] = truth
    return f


def train_triage(f: pd.DataFrame, split_date="2026-03-31"):
    tr = f[f["alert_date"] <= split_date]
    te = f[f["alert_date"] > split_date]
    m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=10,
                                       random_state=42).fit(tr[TRIAGE_FEATURES], tr["sar_filed"])
    p = m.predict_proba(te[TRIAGE_FEATURES])[:, 1]
    te = te.assign(priority=p).sort_values("priority", ascending=False)
    y = te["_true_suspicious"].to_numpy()
    n = len(te)
    k20 = int(np.ceil(0.2 * n))
    cum = np.cumsum(y) / max(y.sum(), 1)
    to90 = int(np.searchsorted(cum, 0.9) + 1) if y.sum() else 0
    metrics = {"alerts_test": n, "truly_suspicious": int(y.sum()), "false_positive_rate_of_rules": round(1 - y.mean(), 3),
               "suspicious_found_in_top_20pct": round(float(y[:k20].sum() / max(y.sum(), 1)), 3),
               "alerts_to_review_for_90pct_recall": to90,
               "workload_reduction_at_90pct_recall": round(1 - to90 / max(n, 1), 3)}
    return m, te, metrics


def scenario_coverage(alerts: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    typ = cust.set_index("customer_id")["_aml_typology"]
    rows = []
    for t in sorted(typ.dropna().unique()):
        who = set(typ[typ == t].index)
        hit = set(alerts["customer_id"]) & who
        rows.append({"typology": t, "customers": len(who), "detected_by_any_scenario": len(hit),
                     "coverage": round(len(hit) / len(who), 3)})
    per = alerts.assign(true=alerts["customer_id"].map(typ).notna()).groupby("scenario_name").agg(
        alerts=("true", "size"), precision=("true", "mean")).round(3)
    return pd.DataFrame(rows), per
