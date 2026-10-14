"""Financial crime operations dashboard: fraud alert queue with reasons, AML triage queue, KPIs.

    streamlit run dashboard.py      (run `make pipeline` first)
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Financial Crime Monitor", page_icon="🛡️", layout="wide")
OUT = Path("outputs")
if not (OUT / "metrics.json").exists():
    st.error("Run `make pipeline` first.")
    st.stop()

m = json.loads((OUT / "metrics.json").read_text())
fq = pd.read_csv(OUT / "fraud_alert_queue.csv")
aq = pd.read_csv(OUT / "aml_alert_queue.csv")
f, t = m["fraud"], m["aml_triage"]

st.title("🛡️ Financial Crime Monitor")
st.caption("Card fraud detection and AML transaction monitoring · retail bank, UAE · test period May–June 2026")

k = st.columns(5)
k[0].metric("Fraud caught", f"{f['model']['fraud_caught']}/{f['model']['fraud_total']}",
            f"rules: {f['rules_baseline']['fraud_caught']}")
k[1].metric("Fraud loss avoided", f"AED {(f['no_action']['fraud_value_missed_aed'] - f['model']['fraud_value_missed_aed']) / 1e3:,.0f}K")
k[2].metric("Alert precision", f"{f['model']['precision']:.0%}", f"rules: {f['rules_baseline']['precision']:.0%}")
k[3].metric("AML false positives (rules)", f"{t['false_positive_rate_of_rules']:.0%}")
k[4].metric("AML workload saved", f"{t['workload_reduction_at_90pct_recall']:.0%}", "at 90% recall")

tab1, tab2, tab3 = st.tabs(["💳 Fraud alert queue", "🏦 AML triage queue", "📄 Full report"])
with tab1:
    st.markdown("Highest-risk card transactions first, with **reason codes** so analysts know what to check.")
    show = fq.copy()
    show["confirmed fraud"] = show.pop("is_fraud").map({1: "yes", 0: "no"})
    st.dataframe(show.style.format({"score": "{:.3f}", "amount_aed": "AED {:,.2f}"}), use_container_width=True,
                 height=480, hide_index=True)
with tab2:
    st.markdown("Rule alerts re-ordered by the triage model. Work from the top; the bottom of the queue is mostly "
                "legitimate activity (property purchases, business cash, family remittances).")
    sc = st.multiselect("Scenario", sorted(aq["scenario_name"].unique()), default=sorted(aq["scenario_name"].unique()))
    view = aq[aq["scenario_name"].isin(sc)].copy()
    view["truly suspicious"] = view.pop("_true_suspicious").map({1: "yes", 0: "no"})
    st.dataframe(view.style.format({"priority": "{:.2f}", "amount_aed": "AED {:,.0f}", "foreign_out_share": "{:.0%}",
                                    "anomaly_score": "{:.2f}"}), use_container_width=True, height=420, hide_index=True)
    ranked = aq.reset_index(drop=True)
    curve = (ranked["_true_suspicious"].cumsum() / ranked["_true_suspicious"].sum()).rename("share of suspicious cases found")
    curve.index = (curve.index + 1) / len(curve)
    st.markdown("**Cumulative suspicious cases found vs share of queue reviewed**")
    st.line_chart(curve, height=240)
with tab3:
    st.markdown((OUT / "report.md").read_text())
