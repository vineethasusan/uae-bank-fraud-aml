"""Real-time card fraud scoring.

Features describe how unusual a transaction is *for this customer* at the moment it
happens: amount vs their history, new country / device / merchant, velocity in the last
hour and day, card-testing bursts, overseas ATM shortly after a local one. Only past
transactions are used for every feature (no look-ahead), so the same code can score
transactions live.

Validation is by time (train Jan-Mar, choose the alert threshold on April, test May-June)
and the threshold is chosen on cost: investigating an alert costs money, missed fraud
costs its amount.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REVIEW_COST = 25.0        # AED per alert reviewed by an analyst (incl. customer call)
TRAIN_END, VALID_END = "2026-03-31 23:59", "2026-04-30 23:59"

FEATURES = ["amount_aed", "log_amount", "amount_vs_median", "amount_vs_max", "hour", "is_night", "is_weekend",
            "channel_code", "mcc_code", "is_foreign", "new_country", "new_device", "new_merchant",
            "txn_1h", "txn_24h", "amount_24h", "small_txn_1h", "countries_24h", "secs_since_prev",
            "foreign_atm_after_local_24h", "history_len", "segment_code", "account_age_months"]
REASONS = {
    "amount_vs_median": "Amount far above this customer's usual spend",
    "amount_vs_max": "Larger than anything this customer has spent before",
    "new_country": "First transaction in this country",
    "new_device": "New device for online payments",
    "small_txn_1h": "Burst of tiny online payments (card testing)",
    "txn_1h": "Many transactions in the last hour",
    "amount_24h": "High total spend in the last 24 hours",
    "foreign_atm_after_local_24h": "Overseas ATM soon after a local ATM (possible skimming)",
    "countries_24h": "Several countries within a day",
    "is_foreign": "Overseas transaction",
    "new_merchant": "First payment at this merchant",
    "is_night": "Unusual hour",
    "mcc_code": "High-risk merchant category",
    "secs_since_prev": "Unusually quick succession of payments",
}


def _per_customer(g: pd.DataFrame) -> pd.DataFrame:
    ts = g["ts"].to_numpy("datetime64[s]").astype(np.int64)
    amt = g["amount_aed"].to_numpy(float)
    n = len(g)
    out = {k: np.zeros(n) for k in ("amount_vs_median", "amount_vs_max", "new_country", "new_device", "new_merchant",
                                    "txn_1h", "txn_24h", "amount_24h", "small_txn_1h", "countries_24h",
                                    "secs_since_prev", "foreign_atm_after_local_24h", "history_len")}
    seen_c, seen_d, seen_m = set(), set(), set()
    countries = g["country"].to_numpy()
    devices = g["device_id"].to_numpy()
    merchants = g["merchant_id"].to_numpy()
    channel = g["channel"].to_numpy()
    lo1 = np.searchsorted(ts, ts - 3600, side="left")
    lo24 = np.searchsorted(ts, ts - 86400, side="left")
    last_local_atm = -10**12
    for i in range(n):
        hist = amt[max(0, i - 60):i]
        out["history_len"][i] = i
        med = np.median(hist) if len(hist) else np.nan
        out["amount_vs_median"][i] = amt[i] / med if len(hist) >= 5 else np.nan
        out["amount_vs_max"][i] = amt[i] / hist.max() if len(hist) >= 5 else np.nan
        out["new_country"][i] = float(i >= 5 and countries[i] not in seen_c)
        out["new_device"][i] = float(i >= 5 and isinstance(devices[i], str) and devices[i] not in seen_d)
        out["new_merchant"][i] = float(merchants[i] not in seen_m)
        a1, a24 = lo1[i], lo24[i]
        out["txn_1h"][i] = i - a1
        out["txn_24h"][i] = i - a24
        out["amount_24h"][i] = amt[a24:i].sum()
        out["small_txn_1h"][i] = (amt[a1:i] < 10).sum()
        out["countries_24h"][i] = len(set(countries[a24:i + 1]))
        out["secs_since_prev"][i] = ts[i] - ts[i - 1] if i else 1e7
        if channel[i] == "atm":
            if countries[i] == "AE":
                last_local_atm = ts[i]
            elif ts[i] - last_local_atm < 86400:
                out["foreign_atm_after_local_24h"][i] = 1
        seen_c.add(countries[i])
        if isinstance(devices[i], str):
            seen_d.add(devices[i])
        seen_m.add(merchants[i])
    return pd.DataFrame(out, index=g.index)


def build_features(cards: pd.DataFrame, customers: pd.DataFrame) -> pd.DataFrame:
    df = cards.copy()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.sort_values(["customer_id", "ts"])
    parts = [_per_customer(g) for _, g in df.groupby("customer_id", sort=False)]
    df = df.join(pd.concat(parts))
    df["log_amount"] = np.log1p(df["amount_aed"])
    df["hour"] = df["ts"].dt.hour
    df["is_night"] = df["hour"].between(0, 5).astype(int)
    df["is_weekend"] = df["ts"].dt.dayofweek.isin([5, 6]).astype(int)
    df["is_foreign"] = (df["country"] != "AE").astype(int)
    df["channel_code"] = df["channel"].map({"pos": 0, "ecom": 1, "atm": 2})
    df["mcc_code"] = df["mcc"].astype("category").cat.codes
    c = customers.set_index("customer_id")
    df["segment_code"] = df["customer_id"].map(c["segment"].map({"Mass": 0, "Affluent": 1, "Private": 2}))
    df["account_age_months"] = df["customer_id"].map(c["account_age_months"])
    return df.sort_values("ts").reset_index(drop=True)


def rules_baseline(f: pd.DataFrame) -> np.ndarray:
    """Typical legacy rules: big foreign online payment, or many payments in an hour, or overseas ATM."""
    return (((f["amount_aed"] > 5000) & (f["is_foreign"] == 1) & (f["channel_code"] == 1))
            | (f["txn_1h"] >= 4) | ((f["channel_code"] == 2) & (f["is_foreign"] == 1) & (f["amount_aed"] >= 1500))
            ).astype(int).to_numpy()


def make_model(seed=42):
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                                          l2_regularization=1.0, random_state=seed)


def cost_of(y, alert, amount) -> dict:
    missed = float(amount[(y == 1) & (alert == 0)].sum())
    caught = float(amount[(y == 1) & (alert == 1)].sum())
    n_alerts = int(alert.sum())
    return {"alerts": n_alerts, "fraud_caught": int(((y == 1) & (alert == 1)).sum()), "fraud_total": int(y.sum()),
            "fraud_value_caught_aed": round(caught), "fraud_value_missed_aed": round(missed),
            "precision": round(float(((y == 1) & (alert == 1)).sum() / max(n_alerts, 1)), 3),
            "total_cost_aed": round(missed + n_alerts * REVIEW_COST)}


def best_threshold(y, p, amount) -> float:
    grid = np.unique(np.quantile(p, np.linspace(0.95, 0.99995, 300)))
    costs = [cost_of(y, (p >= t).astype(int), amount)["total_cost_aed"] for t in grid]
    return float(grid[int(np.argmin(costs))])


def train_and_evaluate(feat: pd.DataFrame) -> dict:
    tr = feat[feat["ts"] <= TRAIN_END]
    va = feat[(feat["ts"] > TRAIN_END) & (feat["ts"] <= VALID_END)]
    te = feat[feat["ts"] > VALID_END]
    m = make_model().fit(tr[FEATURES], tr["is_fraud"])
    thr = best_threshold(va["is_fraud"].to_numpy(), m.predict_proba(va[FEATURES])[:, 1], va["amount_aed"].to_numpy())
    # refit on train + validation, keep the threshold
    trva = feat[feat["ts"] <= VALID_END]
    m = make_model().fit(trva[FEATURES], trva["is_fraud"])
    y, amount = te["is_fraud"].to_numpy(), te["amount_aed"].to_numpy()
    p = m.predict_proba(te[FEATURES])[:, 1]
    model_res = cost_of(y, (p >= thr).astype(int), amount)
    rules_res = cost_of(y, rules_baseline(te), amount)
    no_action = cost_of(y, np.zeros_like(y), amount)
    by_type = (te.assign(alert=(p >= thr).astype(int)).query("is_fraud == 1")
               .groupby("_fraud_type")["alert"].mean().round(3).to_dict())
    return {"model": m, "threshold": thr, "test": te.assign(score=p, alert=(p >= thr).astype(int)),
            "metrics": {"roc_auc": round(roc_auc_score(y, p), 4), "pr_auc": round(average_precision_score(y, p), 4),
                        "test_transactions": int(len(te)), "fraud_rate": round(float(y.mean()), 5),
                        "model": model_res, "rules_baseline": rules_res, "no_action": no_action,
                        "detection_rate_by_fraud_type": by_type}}


def reason_codes(model, row: pd.Series, reference: pd.DataFrame, top: int = 3) -> list[str]:
    """Which features push this transaction's score up most (swap each for a typical value)."""
    base = row[FEATURES].to_frame().T.astype(float)
    p0 = model.predict_proba(base)[0, 1]
    med = reference[FEATURES].median()
    deltas = []
    for f in REASONS:
        alt = base.copy()
        alt[f] = med[f]
        deltas.append((p0 - model.predict_proba(alt)[0, 1], f))
    return [REASONS[f] for d, f in sorted(deltas, reverse=True)[:top] if d > 0.01]
