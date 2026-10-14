"""Synthetic retail-banking data with labelled card fraud and money-laundering typologies.

Card transactions (6 months, 3,000 customers) with four fraud patterns:
  * card testing      - bursts of tiny online payments, then a large one
  * account takeover  - new device + new country + unusually large online spend
  * skimming          - ATM withdrawals abroad shortly after a domestic ATM use
  * merchant collusion- repeated high-value POS payments at one risky merchant

Account transfers & cash (same customers) with AML typologies:
  * structuring       - several cash deposits just under the reporting threshold
  * money mule        - many inbound transfers from unrelated senders, quickly forwarded abroad
  * rapid pass-through- large inflow leaves the account within 48 hours
  * high-risk corridor- repeated transfers to high-risk jurisdictions

Labels: `is_fraud` on card transactions; `_aml_typology` on customers (hidden ground truth),
plus historical investigator outcomes on past AML alerts for training the triage model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

START, END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-30 23:59")
REPORTING_THRESHOLD = 55_000           # AED, configurable cash-reporting threshold used by the scenario

MCC = {  # merchant category: (typical amount AED, channel weights pos/ecom/atm)
    "Grocery": (180, (0.95, 0.05, 0)), "Restaurants": (120, (0.8, 0.2, 0)), "Fuel": (110, (1, 0, 0)),
    "Fashion": (450, (0.6, 0.4, 0)), "Electronics": (900, (0.5, 0.5, 0)), "Travel": (2200, (0.1, 0.9, 0)),
    "Utilities": (600, (0, 1, 0)), "Online Subscriptions": (45, (0, 1, 0)), "Cash Withdrawal": (800, (0, 0, 1)),
    "Healthcare": (350, (0.9, 0.1, 0)), "Gold & Jewellery": (2500, (1, 0, 0)),
}
SEGMENTS = {"Mass": (0.6, 0.7), "Affluent": (0.3, 1.8), "Private": (0.1, 4.5)}   # share, spend multiplier
FOREIGN = ["GB", "IN", "PK", "PH", "EG", "SA", "US", "TR", "TH", "FR"]
HIGH_RISK = ["XA", "XB", "XC"]          # placeholder codes for high-risk jurisdictions
RISKY_MERCHANTS = [f"M{9000 + i}" for i in range(6)]


def _customers(rng, n):
    seg_names = list(SEGMENTS)
    seg = rng.choice(seg_names, n, p=[SEGMENTS[s][0] for s in seg_names])
    return pd.DataFrame({
        "customer_id": [f"CU{i + 1:05d}" for i in range(n)],
        "segment": seg,
        "home_emirate": rng.choice(["Dubai", "Abu Dhabi", "Sharjah", "Ajman", "RAK"], n, p=[0.5, 0.25, 0.15, 0.06, 0.04]),
        "nationality_region": rng.choice(["GCC", "South Asia", "Europe", "Arab (other)", "Southeast Asia", "Other"], n,
                                         p=[0.15, 0.35, 0.12, 0.18, 0.12, 0.08]),
        "account_age_months": rng.integers(1, 180, n),
        "monthly_salary_aed": np.round(rng.lognormal(9.4, 0.6, n) * np.array([SEGMENTS[s][1] for s in seg]), -2),
        "occupation_risk": rng.choice(["low", "medium", "high"], n, p=[0.75, 0.2, 0.05]),
    })


def card_transactions(rng, cust: pd.DataFrame, fraud_rate_customers=0.04):
    rows = []
    tid = 0
    seconds = int((END - START).total_seconds())
    mcc_names = list(MCC)
    for c in cust.itertuples():
        mult = SEGMENTS[c.segment][1]
        n = int(rng.poisson(45 * 6 * min(mult, 2.5) ** 0.5))
        habits = rng.dirichlet(np.ones(len(mcc_names)) * 0.6)
        device = f"D{rng.integers(1e8):08d}"
        new_phone_at = START + pd.Timedelta(days=int(rng.integers(20, 170))) if rng.random() < 0.25 else None
        trips = []                                  # legitimate travel: looks "fraudy" but isn't
        for _ in range(rng.poisson(0.5 * mult ** 0.5)):
            a = START + pd.Timedelta(days=int(rng.integers(5, 170)))
            trips.append((a, a + pd.Timedelta(days=int(rng.integers(3, 12))), rng.choice(FOREIGN)))
        ts = np.sort(rng.integers(0, seconds, n))
        for s in ts:
            m = mcc_names[rng.choice(len(mcc_names), p=habits)]
            base, ch = MCC[m]
            channel = ["pos", "ecom", "atm"][rng.choice(3, p=ch)]
            amt = round(float(rng.lognormal(np.log(base * mult ** 0.6), 0.6)), 2)
            t = START + pd.Timedelta(seconds=int(s))
            # people sleep: shift night transactions to daytime mostly
            if t.hour < 6 and rng.random() < 0.85:
                t += pd.Timedelta(hours=int(rng.integers(7, 14)))
            country = "AE" if rng.random() > 0.04 * mult ** 0.5 else rng.choice(FOREIGN)
            for a, b, ctry in trips:
                if a <= t <= b:
                    country = ctry
                    if channel == "ecom" and rng.random() < 0.5:
                        channel, m = "pos", rng.choice(["Restaurants", "Fashion", "Travel", "Cash Withdrawal"])
                        if m == "Cash Withdrawal":
                            channel = "atm"
                        amt = round(float(rng.lognormal(np.log(MCC[m][0] * mult ** 0.6), 0.7)), 2)
            dev = device
            if new_phone_at is not None and t >= new_phone_at:
                dev = device + "N"
            tid += 1
            rows.append((tid, c.customer_id, t, m, f"M{rng.integers(1, 3000):04d}", channel, amt, country,
                         dev if channel == "ecom" else None, 0, None))

        if rng.random() < fraud_rate_customers:
            tid = _inject_fraud(rng, rows, tid, c, mult)

    df = pd.DataFrame(rows, columns=["txn_id", "customer_id", "ts", "mcc", "merchant_id", "channel", "amount_aed",
                                     "country", "device_id", "is_fraud", "_fraud_type"])
    return df.sort_values("ts").reset_index(drop=True)


def _inject_fraud(rng, rows, tid, c, mult):
    kind = rng.choice(["card_testing", "account_takeover", "skimming", "merchant_collusion"], p=[0.3, 0.3, 0.25, 0.15])
    t = START + pd.Timedelta(seconds=int(rng.integers(86400 * 20, int((END - START).total_seconds()) - 86400 * 3)))
    if kind == "card_testing":
        dev = f"D{rng.integers(1e8):08d}"
        for i in range(int(rng.integers(4, 9))):
            tid += 1
            rows.append((tid, c.customer_id, t + pd.Timedelta(minutes=int(i * rng.integers(1, 6))), "Online Subscriptions",
                         f"M{rng.integers(1, 3000):04d}", "ecom", round(float(rng.uniform(1, 9)), 2),
                         rng.choice(FOREIGN), dev, 1, kind))
        tid += 1
        rows.append((tid, c.customer_id, t + pd.Timedelta(minutes=45), "Electronics", f"M{rng.integers(1, 3000):04d}",
                     "ecom", round(float(rng.uniform(1500, 6000)), 2), rng.choice(FOREIGN), dev, 1, kind))
    elif kind == "account_takeover":
        dev = f"D{rng.integers(1e8):08d}"
        country = rng.choice(FOREIGN)
        for i in range(int(rng.integers(2, 5))):
            tid += 1
            rows.append((tid, c.customer_id, t + pd.Timedelta(minutes=int(i * rng.integers(5, 40))),
                         rng.choice(["Electronics", "Travel", "Gold & Jewellery"]), f"M{rng.integers(1, 3000):04d}", "ecom",
                         round(float(rng.uniform(2500, 12000) * mult ** 0.5), 2), country, dev, 1, kind))
    elif kind == "skimming":
        country = rng.choice(FOREIGN)
        for i in range(int(rng.integers(2, 5))):
            tid += 1
            rows.append((tid, c.customer_id, t + pd.Timedelta(hours=int(i * rng.integers(1, 6))), "Cash Withdrawal",
                         f"M{rng.integers(1, 3000):04d}", "atm", float(rng.choice([1000, 1500, 2000, 2500])), country,
                         None, 1, kind))
    else:  # merchant collusion
        merchant = rng.choice(RISKY_MERCHANTS)
        for i in range(int(rng.integers(3, 6))):
            tid += 1
            rows.append((tid, c.customer_id, t + pd.Timedelta(days=int(i * rng.integers(1, 4)), hours=int(rng.integers(0, 6))),
                         "Gold & Jewellery", merchant, "pos", round(float(rng.uniform(4000, 15000)), 2), "AE", None, 1, kind))
    return tid


def account_activity(rng, cust: pd.DataFrame, aml_rate=0.03):
    rows = []
    tid = 0
    days = (END - START).days
    typology = {}
    for c in cust.itertuples():
        sal_day = int(rng.integers(25, 29))
        # salary + routine outflows
        for m in range(6):
            month = START + pd.DateOffset(months=m)
            tid += 1
            rows.append((tid, c.customer_id, month + pd.Timedelta(days=sal_day - 1, hours=9), "salary_in", "credit",
                         c.monthly_salary_aed, "EMPLOYER", "AE"))
            for _ in range(int(rng.integers(2, 6))):
                tid += 1
                rows.append((tid, c.customer_id, month + pd.Timedelta(days=int(rng.integers(0, 28)), hours=int(rng.integers(8, 22))),
                             "transfer_out", "debit", round(float(c.monthly_salary_aed * rng.uniform(0.05, 0.3)), 2),
                             f"EXT{rng.integers(1, 20000):05d}", "AE" if rng.random() < 0.7 else rng.choice(FOREIGN)))
            if rng.random() < 0.3:
                tid += 1
                rows.append((tid, c.customer_id, month + pd.Timedelta(days=int(rng.integers(0, 28)), hours=11), "cash_deposit",
                             "credit", round(float(rng.uniform(500, 8000)), 2), "BRANCH", "AE"))
            if rng.random() < 0.15:   # occasional legitimate large inflow (bonus, property sale, gift)
                tid += 1
                rows.append((tid, c.customer_id, month + pd.Timedelta(days=int(rng.integers(0, 28)), hours=10),
                             "transfer_in", "credit", round(float(c.monthly_salary_aed * rng.uniform(2, 8)), 2),
                             f"EXT{rng.integers(1, 20000):05d}", "AE"))
        if rng.random() < aml_rate:
            kind = rng.choice(["structuring", "money_mule", "rapid_pass_through", "high_risk_corridor"])
            typology[c.customer_id] = kind
            tid = _inject_aml(rng, rows, tid, c, kind, days)
        elif rng.random() < 0.05:
            # legitimate behaviour that looks like a typology: the false positives investigators deal with
            tid = _inject_lookalike(rng, rows, tid, c, days)
    df = pd.DataFrame(rows, columns=["txn_id", "customer_id", "ts", "type", "direction", "amount_aed", "counterparty",
                                     "counterparty_country"])
    return df.sort_values("ts").reset_index(drop=True), typology


def _inject_aml(rng, rows, tid, c, kind, days):
    t0 = START + pd.Timedelta(days=int(rng.integers(10, days - 20)))
    if kind == "structuring":
        for i in range(int(rng.integers(4, 9))):
            tid += 1
            rows.append((tid, c.customer_id, t0 + pd.Timedelta(days=int(i * rng.integers(1, 4)), hours=int(rng.integers(9, 17))),
                         "cash_deposit", "credit", round(float(REPORTING_THRESHOLD * rng.uniform(0.85, 0.99)), -2), "BRANCH", "AE"))
    elif kind == "money_mule":
        total = 0.0
        for i in range(int(rng.integers(8, 20))):
            amt = round(float(rng.uniform(2000, 15000)), 2)
            total += amt
            tid += 1
            rows.append((tid, c.customer_id, t0 + pd.Timedelta(hours=int(i * rng.integers(2, 12))), "transfer_in", "credit",
                         amt, f"EXT{rng.integers(20000, 40000):05d}", "AE"))
        for j in range(int(rng.integers(1, 3))):
            tid += 1
            rows.append((tid, c.customer_id, t0 + pd.Timedelta(days=4 + j, hours=10), "transfer_out", "debit",
                         round(total * 0.95 / 2, 2), f"EXT{rng.integers(40000, 41000):05d}", rng.choice(FOREIGN)))
    elif kind == "rapid_pass_through":
        for k in range(int(rng.integers(2, 4))):
            amt = round(float(rng.uniform(150_000, 900_000)), -2)
            tt = t0 + pd.Timedelta(days=int(k * rng.integers(7, 20)))
            tid += 1
            rows.append((tid, c.customer_id, tt, "transfer_in", "credit", amt, f"EXT{rng.integers(41000, 42000):05d}", "AE"))
            tid += 1
            rows.append((tid, c.customer_id, tt + pd.Timedelta(hours=int(rng.integers(6, 40))), "transfer_out", "debit",
                         round(amt * rng.uniform(0.93, 0.99), -2), f"EXT{rng.integers(42000, 43000):05d}", rng.choice(FOREIGN)))
    else:  # high-risk corridor
        for i in range(int(rng.integers(4, 10))):
            tid += 1
            rows.append((tid, c.customer_id, t0 + pd.Timedelta(days=int(i * rng.integers(2, 8))), "transfer_out", "debit",
                         round(float(rng.uniform(8000, 45000)), 2), f"EXT{rng.integers(43000, 43100):05d}", rng.choice(HIGH_RISK)))
    return tid


def _inject_lookalike(rng, rows, tid, c, days):
    t0 = START + pd.Timedelta(days=int(rng.integers(10, days - 20)))
    kind = rng.choice(["property_purchase", "business_cash", "family_support", "group_collection"])
    if kind == "property_purchase":          # mortgage/sale proceeds in, paid to a developer the next day
        amt = round(float(rng.uniform(300_000, 2_000_000)), -3)
        tid += 1
        rows.append((tid, c.customer_id, t0, "transfer_in", "credit", amt, f"EXT{rng.integers(1, 20000):05d}", "AE"))
        tid += 1
        rows.append((tid, c.customer_id, t0 + pd.Timedelta(hours=int(rng.integers(12, 48))), "transfer_out", "debit",
                     round(amt * 0.97, -2), "EXT_DEVELOPER", "AE"))
    elif kind == "business_cash":            # shop owner banking weekly takings
        for i in range(int(rng.integers(3, 5))):
            tid += 1
            rows.append((tid, c.customer_id, t0 + pd.Timedelta(days=int(i * rng.integers(3, 5)), hours=17),
                         "cash_deposit", "credit", round(float(REPORTING_THRESHOLD * rng.uniform(0.8, 0.98)), -2), "BRANCH", "AE"))
    elif kind == "family_support":           # small regular remittances to relatives
        for i in range(int(rng.integers(3, 5))):
            tid += 1
            rows.append((tid, c.customer_id, t0 + pd.Timedelta(days=int(i * rng.integers(5, 9))), "transfer_out", "debit",
                         round(float(rng.uniform(800, 4000)), 2), f"EXT{rng.integers(1, 20000):05d}", rng.choice(HIGH_RISK)))
    else:                                    # collecting money from colleagues for a shared trip, then paying abroad
        total = 0.0
        for i in range(int(rng.integers(8, 14))):
            amt = round(float(rng.uniform(1500, 4000)), 2)
            total += amt
            tid += 1
            rows.append((tid, c.customer_id, t0 + pd.Timedelta(hours=int(i * rng.integers(3, 12))), "transfer_in", "credit",
                         amt, f"EXT{rng.integers(1, 20000):05d}", "AE"))
        tid += 1
        rows.append((tid, c.customer_id, t0 + pd.Timedelta(days=5), "transfer_out", "debit", round(total * 0.98, 2),
                     "EXT_TRAVEL_AGENT", rng.choice(FOREIGN)))
    return tid


def generate(n_customers: int = 3000, seed: int = 42):
    rng = np.random.default_rng(seed)
    cust = _customers(rng, n_customers)
    cards = card_transactions(rng, cust)
    acct, typ = account_activity(rng, cust)
    cust["_aml_typology"] = cust["customer_id"].map(typ)
    return cust, cards, acct


if __name__ == "__main__":
    from pathlib import Path

    Path("data").mkdir(exist_ok=True)
    cust, cards, acct = generate()
    cust.to_csv("data/customers.csv", index=False)
    cards.to_csv("data/card_transactions.csv.gz", index=False)
    acct.to_csv("data/account_transactions.csv.gz", index=False)
    print(f"{len(cust):,} customers | {len(cards):,} card txns ({cards.is_fraud.sum()} fraud, "
          f"{cards.is_fraud.mean():.2%}) | {len(acct):,} account txns | "
          f"{cust._aml_typology.notna().sum()} customers with AML typologies")
