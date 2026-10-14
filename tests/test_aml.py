import pandas as pd

from fincrime import aml
from fincrime.data import REPORTING_THRESHOLD


def _cust(salary=10_000):
    return pd.DataFrame({"customer_id": ["C1"], "monthly_salary_aed": [salary], "account_age_months": [24],
                         "occupation_risk": ["low"], "segment": ["Mass"], "_aml_typology": [None]})


def _acct(rows):
    return pd.DataFrame(rows, columns=["customer_id", "ts", "type", "direction", "amount_aed", "counterparty",
                                       "counterparty_country"]).assign(ts=lambda d: pd.to_datetime(d["ts"]))


def test_structuring_scenario():
    rows = [("C1", f"2026-03-0{d} 11:00", "cash_deposit", "credit", REPORTING_THRESHOLD * 0.95, "BRANCH", "AE")
            for d in (1, 4, 7)]
    a = aml.run_scenarios(_acct(rows), _cust())
    assert "S1" in set(a["scenario"])


def test_two_deposits_are_not_structuring():
    rows = [("C1", f"2026-03-0{d} 11:00", "cash_deposit", "credit", REPORTING_THRESHOLD * 0.95, "BRANCH", "AE")
            for d in (1, 4)]
    assert "S1" not in set(aml.run_scenarios(_acct(rows), _cust())["scenario"])


def test_pass_through_scenario():
    rows = [("C1", "2026-03-01 10:00", "transfer_in", "credit", 500_000, "EXT1", "AE"),
            ("C1", "2026-03-02 15:00", "transfer_out", "debit", 480_000, "EXT2", "GB")]
    assert "S2" in set(aml.run_scenarios(_acct(rows), _cust())["scenario"])


def test_triage_finds_suspicious_cases_early(bank):
    cust, _, acct = bank
    alerts = aml.run_scenarios(acct, cust)
    f = aml.build_alert_features(alerts, acct, cust)
    _, _, met = aml.train_triage(f)
    if met["truly_suspicious"]:
        assert met["suspicious_found_in_top_20pct"] >= 0.6
