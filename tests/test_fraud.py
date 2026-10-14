import numpy as np
import pandas as pd

from fincrime import fraud


def _txns(rows):
    df = pd.DataFrame(rows, columns=["ts", "amount_aed", "country", "channel", "device_id", "merchant_id"])
    df["ts"] = pd.to_datetime(df["ts"])
    df["customer_id"] = "C1"
    return df


def test_velocity_and_card_testing_features():
    rows = [(f"2026-05-01 10:{m:02d}", a, "AE", "ecom", "D1", f"M{m}") for m, a in
            [(0, 3.0), (5, 4.0), (9, 2.5), (20, 5000.0)]]
    out = fraud._per_customer(_txns(rows))
    assert out["txn_1h"].tolist() == [0, 1, 2, 3]
    assert out["small_txn_1h"].iloc[-1] == 3


def test_features_never_look_ahead():
    rows = [(f"2026-05-0{d} 12:00", 100.0 * d, "AE", "pos", None, "M1") for d in range(1, 9)]
    full = fraud._per_customer(_txns(rows))
    part = fraud._per_customer(_txns(rows[:6]))
    assert np.allclose(full.iloc[:6].fillna(-1).to_numpy(), part.fillna(-1).to_numpy())


def test_foreign_atm_after_local():
    rows = [("2026-05-01 09:00", 500, "AE", "atm", None, "A1"), ("2026-05-01 20:00", 2000, "GB", "atm", None, "A2")]
    assert fraud._per_customer(_txns(rows))["foreign_atm_after_local_24h"].tolist() == [0, 1]


def test_model_beats_rules_on_cost(bank):
    cust, cards, _ = bank
    feat = fraud.build_features(cards, cust)
    res = fraud.train_and_evaluate(feat)
    m = res["metrics"]
    assert m["model"]["total_cost_aed"] < m["rules_baseline"]["total_cost_aed"] < m["no_action"]["total_cost_aed"]
