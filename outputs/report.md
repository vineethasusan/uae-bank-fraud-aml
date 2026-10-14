# Financial crime analytics: results

## Card fraud (test period: May–June 2026)

289,939 transactions, fraud rate 0.082%. ROC-AUC 0.9986, PR-AUC 0.9307.

| Approach | Alerts | Fraud caught | Precision | Fraud loss | Total cost (loss + reviews) |
|---|---|---|---|---|---|
| No monitoring | 0 | 0/237 | 0% | AED 1,031,302 | AED 1,031,302 |
| Legacy rules | 1,272 | 112/237 | 9% | AED 430,216 | AED 462,016 |
| ML model (cost-optimised threshold) | 901 | 230/237 | 26% | AED 55,594 | AED 78,119 |

Detection rate by fraud pattern: account takeover 100%, card testing 100%, merchant collusion 80%, skimming 100%

## AML monitoring

**Scenario coverage of known typologies**

| Typology | Customers | Detected | Coverage |
|---|---|---|---|
| high risk corridor | 20 | 20 | 100% |
| money mule | 23 | 22 | 96% |
| rapid pass through | 16 | 16 | 100% |
| structuring | 20 | 20 | 100% |

**Scenario precision** (share of alerts that are truly suspicious)

| Scenario | Alerts | Precision |
|---|---|---|
| Fan-in / money mule | 39 | 56% |
| High-risk corridor | 48 | 42% |
| Income mismatch | 1,536 | 4% |
| Rapid pass-through | 50 | 32% |
| Structuring | 52 | 38% |

**Alert triage (Apr–Jun 2026 alerts, model trained on Jan–Mar decisions)**

- 771 alerts, of which 64 truly suspicious (92% false positives from rules alone)
- Top 20% of the ranked queue contains **95%** of suspicious cases
- 90% of suspicious cases are found after reviewing **125** alerts: a **84% workload reduction**
