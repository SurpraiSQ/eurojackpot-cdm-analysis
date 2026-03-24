# EuroJackpot CDM Analysis

This repository contains an exploratory statistical analysis of EuroJackpot draw data
and a replication of a Compound Dirichlet–Multinomial (CDM)–based modeling approach.

The project focuses on observed number frequencies, deviation from expected values,
volatility over time, and reversion behavior.  
No claims of predictability or guaranteed outcomes are made.

---

## Repository structure

```eurojackpot-cdm-analysis/
├─ README.md
├─ EuroJackpot Analysis.pdf
│
├─ powerbi/
│ └─ EuroJackpot Analysis.pbix
│
├─ model/
│ ├─ data/
│ │ └─ eurojackpot.csv
│ │
│ ├─ src/
│ │ ├─ alpha_mle.py
│ │ └─ backtest.py
│ │
│ └─ outputs/
│ │├─ predictions.csv
│ │├─ state.json
│ │└─ summary.json
```

---

## Power BI report

The primary analytical output is provided as a static PDF:

**EuroJackpot Analysis.pdf**

It contains all final visualizations and does not require any external data connections.
The `.pbix` file is included for reference as a working Power BI model.

---

## Model

The `model/` directory contains:
- input data used for the analysis,
- source code implementing the CDM logic and evaluation,
- saved outputs from a single evaluation run.

The modeling approach follows a deterministic, walk-forward evaluation setup.

---

## Notes

- This project is exploratory and analytical in nature.
- It does not assert that lottery outcomes are predictable.
- Results are presented for study and discussion purposes only.
