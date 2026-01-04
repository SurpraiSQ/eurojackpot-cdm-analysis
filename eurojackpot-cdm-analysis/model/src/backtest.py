# src/run_backtest.py
# v0.2 — CDM with MM or MLE alpha, deterministic top-K
# Run (recommended, no activation needed):
#   .\.venv\Scripts\python.exe -u .\src\run_backtest.py

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from alpha_mle import fit_alpha_mle


# ==========================
# CONFIG — EDIT HERE IF NEEDED
# ==========================

# IMPORTANT: Change this if you want to test another dataset or a different file variant.
DATA_PATH = os.path.join("data", "eurojackpot.csv")

OUTPUT_DIR = "outputs"
PREDICTIONS_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")
SUMMARY_JSON = os.path.join(OUTPUT_DIR, "summary.json")
STATE_JSON = os.path.join(OUTPUT_DIR, "state.json")

DATE_FORMAT = "%d-%m-%Y"  # dd-mm-yyyy

EURO_START_DATE = date(2022, 3, 25)
REPORT_START_DATE = date(2024, 1, 1)
REPORT_LAST_N = 200

WARMUP_MAIN = 200
WARMUP_EURO = 120

RANDOM_SEED = 42

# Alpha mode:
ALPHA_MODE = "mle"     # "mm" or "mle"
MLE_L2 = 0.0           # try 0.0 then 1e-3
MLE_MAXITER = 200      # try 200-500 if needed


# ==========================
# DATA CONTRACT
# ==========================

REQUIRED_COLUMNS = ["date", "main1", "main2", "main3", "main4", "main5", "euro1", "euro2"]

MAIN_MIN, MAIN_MAX, MAIN_K = 1, 50, 5
EURO_MIN, EURO_MAX, EURO_K = 1, 12, 2


@dataclass
class CountsState:
    """Incremental state to allow continuing later."""
    processed_dates: List[str]  # ISO strings
    main_counts: List[int]      # length 50
    euro_counts: List[int]      # length 12 (only euro-era)
    main_draws_seen: int
    euro_draws_seen: int
    main_beta: List[float]      # length 50 (warm start for MLE)
    euro_beta: List[float]      # length 12

    @staticmethod
    def fresh() -> "CountsState":
        return CountsState(
            processed_dates=[],
            main_counts=[0] * MAIN_MAX,
            euro_counts=[0] * EURO_MAX,
            main_draws_seen=0,
            euro_draws_seen=0,
            main_beta=[0.0] * MAIN_MAX,  # beta=0 => alpha=1
            euro_beta=[0.0] * EURO_MAX,
        )


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_json(path: str, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_date_series(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format=DATE_FORMAT, errors="raise").dt.date


def validate_row_numbers(row: pd.Series) -> None:
    mains = [int(row[f"main{i}"]) for i in range(1, 6)]
    euros = [int(row[f"euro{i}"]) for i in range(1, 3)]

    if len(set(mains)) != 5:
        raise ValueError(f"Main numbers are not unique in row date={row['date']}: {mains}")
    if len(set(euros)) != 2:
        raise ValueError(f"Euro numbers are not unique in row date={row['date']}: {euros}")

    if not all(MAIN_MIN <= x <= MAIN_MAX for x in mains):
        raise ValueError(f"Main numbers out of range in row date={row['date']}: {mains}")
    if not all(EURO_MIN <= x <= EURO_MAX for x in euros):
        raise ValueError(f"Euro numbers out of range in row date={row['date']}: {euros}")


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    df = df[REQUIRED_COLUMNS].copy()
    df["date"] = parse_date_series(df["date"])

    df = df.sort_values("date").reset_index(drop=True)

    if df["date"].duplicated().any():
        dups = df.loc[df["date"].duplicated(), "date"].astype(str).tolist()[:10]
        raise ValueError(f"Duplicate dates detected (showing up to 10): {dups}")

    for _, row in df.iterrows():
        validate_row_numbers(row)

    return df


def hits(pred: List[int], actual: List[int]) -> int:
    return len(set(pred).intersection(set(actual)))


def top_k_from_weights(weights: np.ndarray, k: int) -> List[int]:
    idx = np.argsort(-weights)[:k]
    nums = (idx + 1).tolist()
    nums.sort()
    return nums


def random_pick(rng: np.random.Generator, n_max: int, k: int) -> List[int]:
    nums = rng.choice(np.arange(1, n_max + 1), size=k, replace=False)
    return sorted(int(x) for x in nums)


def cdm_mm_predict(counts: np.ndarray, draws_seen: int, k_pick: int) -> List[int]:
    if draws_seen <= 0:
        raise ValueError("draws_seen must be > 0 for prediction")
    alpha = counts / draws_seen
    scores = alpha + counts
    return top_k_from_weights(scores, k_pick)


def frequency_predict(counts: np.ndarray, k_pick: int) -> List[int]:
    return top_k_from_weights(counts.astype(float), k_pick)


def event_flags(main_hits: int, euro_hits: int) -> Dict[str, int]:
    flags: Dict[str, int] = {}
    flags["2+1"] = int(main_hits >= 2 and euro_hits >= 1)
    for mh in [3, 4, 5]:
        for eh in [0, 1, 2]:
            flags[f"{mh}+{eh}"] = int(main_hits >= mh and euro_hits >= eh)
    return flags


def load_state_if_exists(path: str) -> Optional[CountsState]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    return CountsState(
        processed_dates=raw["processed_dates"],
        main_counts=raw["main_counts"],
        euro_counts=raw["euro_counts"],
        main_draws_seen=raw["main_draws_seen"],
        euro_draws_seen=raw["euro_draws_seen"],
        main_beta=raw.get("main_beta", [0.0] * MAIN_MAX),
        euro_beta=raw.get("euro_beta", [0.0] * EURO_MAX),
    )


def main() -> None:
    ensure_output_dir()

    df = load_data(DATA_PATH)

    prev_state = load_state_if_exists(STATE_JSON)
    if prev_state is None:
        state = CountsState.fresh()
        existing_predictions = None
    else:
        state = prev_state
        if os.path.exists(PREDICTIONS_CSV):
            existing_predictions = pd.read_csv(PREDICTIONS_CSV)
        else:
            existing_predictions = None

    rng = np.random.default_rng(RANDOM_SEED)

    main_counts = np.array(state.main_counts, dtype=int)
    euro_counts = np.array(state.euro_counts, dtype=int)

    main_beta = np.array(state.main_beta, dtype=float)
    euro_beta = np.array(state.euro_beta, dtype=float)

    processed_set = set(state.processed_dates)

    rows_out: List[dict] = []
    new_rows = 0

    for _, row in df.iterrows():
        d: date = row["date"]
        d_iso = d.isoformat()

        if d_iso in processed_set:
            continue

        mains_actual = [int(row[f"main{i}"]) for i in range(1, 6)]
        euros_actual = [int(row[f"euro{i}"]) for i in range(1, 3)]

        euro_eligible = d >= EURO_START_DATE

        can_predict_main = state.main_draws_seen >= WARMUP_MAIN
        can_predict_euro = state.euro_draws_seen >= WARMUP_EURO

        # --- CDM predictions ---
        if can_predict_main:
            if ALPHA_MODE == "mle":
                alpha_main, main_beta = fit_alpha_mle(
                    main_counts.astype(float),
                    beta_init=main_beta,
                    l2=MLE_L2,
                    maxiter=MLE_MAXITER,
                )
                pred_main_cdm = top_k_from_weights(alpha_main + main_counts, MAIN_K)
            else:
                pred_main_cdm = cdm_mm_predict(main_counts, state.main_draws_seen, MAIN_K)
        else:
            pred_main_cdm = []

        if can_predict_euro and euro_eligible:
            if ALPHA_MODE == "mle":
                alpha_euro, euro_beta = fit_alpha_mle(
                    euro_counts.astype(float),
                    beta_init=euro_beta,
                    l2=MLE_L2,
                    maxiter=MLE_MAXITER,
                )
                pred_euro_cdm = top_k_from_weights(alpha_euro + euro_counts, EURO_K)
            else:
                pred_euro_cdm = cdm_mm_predict(euro_counts, state.euro_draws_seen, EURO_K)
        else:
            pred_euro_cdm = []

        # baselines
        pred_main_freq = frequency_predict(main_counts, MAIN_K) if can_predict_main else []
        pred_euro_freq = frequency_predict(euro_counts, EURO_K) if (can_predict_euro and euro_eligible) else []

        pred_main_rand = random_pick(rng, MAIN_MAX, MAIN_K) if can_predict_main else []
        pred_euro_rand = random_pick(rng, EURO_MAX, EURO_K) if (can_predict_euro and euro_eligible) else []

        # evaluation only if we predicted both parts
        if can_predict_main and (can_predict_euro and euro_eligible):
            main_hits_cdm = hits(pred_main_cdm, mains_actual)
            euro_hits_cdm = hits(pred_euro_cdm, euros_actual)

            main_hits_freq = hits(pred_main_freq, mains_actual)
            euro_hits_freq = hits(pred_euro_freq, euros_actual)

            main_hits_rand = hits(pred_main_rand, mains_actual)
            euro_hits_rand = hits(pred_euro_rand, euros_actual)

            flags_cdm = event_flags(main_hits_cdm, euro_hits_cdm)
            flags_freq = event_flags(main_hits_freq, euro_hits_freq)
            flags_rand = event_flags(main_hits_rand, euro_hits_rand)
        else:
            main_hits_cdm = euro_hits_cdm = -1
            main_hits_freq = euro_hits_freq = -1
            main_hits_rand = euro_hits_rand = -1
            proto = event_flags(0, 0).keys()
            flags_cdm = {k: -1 for k in proto}
            flags_freq = {k: -1 for k in proto}
            flags_rand = {k: -1 for k in proto}

        out = {
            "date": d_iso,
            "main_actual": " ".join(map(str, sorted(mains_actual))),
            "euro_actual": " ".join(map(str, sorted(euros_actual))),

            "main_pred_cdm": " ".join(map(str, pred_main_cdm)) if pred_main_cdm else "",
            "euro_pred_cdm": " ".join(map(str, pred_euro_cdm)) if pred_euro_cdm else "",

            "main_pred_freq": " ".join(map(str, pred_main_freq)) if pred_main_freq else "",
            "euro_pred_freq": " ".join(map(str, pred_euro_freq)) if pred_euro_freq else "",

            "main_pred_rand": " ".join(map(str, pred_main_rand)) if pred_main_rand else "",
            "euro_pred_rand": " ".join(map(str, pred_euro_rand)) if pred_euro_rand else "",

            "main_hits_cdm": main_hits_cdm,
            "euro_hits_cdm": euro_hits_cdm,
            "main_hits_freq": main_hits_freq,
            "euro_hits_freq": euro_hits_freq,
            "main_hits_rand": main_hits_rand,
            "euro_hits_rand": euro_hits_rand,

            "can_evaluate": int(can_predict_main and can_predict_euro and euro_eligible),
            "euro_eligible": int(euro_eligible),
        }

        for k, v in flags_cdm.items():
            out[f"cdm_{k}"] = v
        for k, v in flags_freq.items():
            out[f"freq_{k}"] = v
        for k, v in flags_rand.items():
            out[f"rand_{k}"] = v

        rows_out.append(out)
        new_rows += 1

        # update counts AFTER evaluation
        for x in mains_actual:
            main_counts[x - 1] += 1
        state.main_draws_seen += 1

        if euro_eligible:
            for x in euros_actual:
                euro_counts[x - 1] += 1
            state.euro_draws_seen += 1

        state.processed_dates.append(d_iso)
        processed_set.add(d_iso)

    df_new = pd.DataFrame(rows_out)
    if existing_predictions is not None and not existing_predictions.empty:
        combined = pd.concat([existing_predictions, df_new], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        df_pred = combined
    else:
        df_pred = df_new.sort_values("date").reset_index(drop=True)

    df_pred.to_csv(PREDICTIONS_CSV, index=False)

    # persist betas
    state.main_counts = main_counts.tolist()
    state.euro_counts = euro_counts.tolist()
    state.main_beta = main_beta.tolist()
    state.euro_beta = euro_beta.tolist()

    save_json(STATE_JSON, {
        "processed_dates": state.processed_dates,
        "main_counts": state.main_counts,
        "euro_counts": state.euro_counts,
        "main_draws_seen": state.main_draws_seen,
        "euro_draws_seen": state.euro_draws_seen,
        "main_beta": state.main_beta,
        "euro_beta": state.euro_beta,
        "config": {
            "data_path": DATA_PATH,
            "date_format": DATE_FORMAT,
            "euro_start_date": EURO_START_DATE.isoformat(),
            "report_start_date": REPORT_START_DATE.isoformat(),
            "report_last_n": REPORT_LAST_N,
            "warmup_main": WARMUP_MAIN,
            "warmup_euro": WARMUP_EURO,
            "random_seed": RANDOM_SEED,
            "selection": "deterministic_topK",
            "alpha_mode": ALPHA_MODE,
            "mle_l2": MLE_L2,
            "mle_maxiter": MLE_MAXITER,
            "window": "expanding",
        }
    })

    # summary
    if df_pred.empty:
        save_json(SUMMARY_JSON, {"message": "No rows processed."})
        print("No rows processed.")
        return

    df_eval = df_pred[df_pred["can_evaluate"] == 1].copy()
    df_eval["date"] = pd.to_datetime(df_eval["date"]).dt.date

    df_report = df_eval[df_eval["date"] >= REPORT_START_DATE].copy()
    df_last_n = df_eval.tail(REPORT_LAST_N).copy()

    def aggregate_events(df_slice: pd.DataFrame, prefix: str) -> Dict[str, float]:
        if df_slice.empty:
            return {}
        event_cols = [c for c in df_slice.columns if c.startswith(prefix)]
        return {c: float(df_slice[c].mean()) for c in sorted(event_cols)}

    def aggregate_hits(df_slice: pd.DataFrame) -> Dict[str, float]:
        if df_slice.empty:
            return {}
        return {
            "avg_main_hits_cdm": float(df_slice["main_hits_cdm"].mean()),
            "avg_euro_hits_cdm": float(df_slice["euro_hits_cdm"].mean()),
            "avg_main_hits_freq": float(df_slice["main_hits_freq"].mean()),
            "avg_euro_hits_freq": float(df_slice["euro_hits_freq"].mean()),
            "avg_main_hits_rand": float(df_slice["main_hits_rand"].mean()),
            "avg_euro_hits_rand": float(df_slice["euro_hits_rand"].mean()),
            "n_draws": int(len(df_slice)),
            "from": df_slice["date"].min().isoformat(),
            "to": df_slice["date"].max().isoformat(),
        }

    summary = {
        "config": {
            "data_path": DATA_PATH,
            "date_format": DATE_FORMAT,
            "euro_start_date": EURO_START_DATE.isoformat(),
            "report_start_date": REPORT_START_DATE.isoformat(),
            "report_last_n": REPORT_LAST_N,
            "warmup_main": WARMUP_MAIN,
            "warmup_euro": WARMUP_EURO,
            "random_seed": RANDOM_SEED,
            "selection": "deterministic_topK",
            "alpha_mode": ALPHA_MODE,
            "mle_l2": MLE_L2,
            "mle_maxiter": MLE_MAXITER,
            "window": "expanding",
        },
        "processed": {
            "total_rows_in_data": int(len(df)),
            "new_rows_processed_this_run": int(new_rows),
            "total_predictions_rows": int(len(df_pred)),
            "evaluatable_rows_total": int(len(df_eval)),
        },
        "report_window": {
            "hits": aggregate_hits(df_report),
            "events_cdm": aggregate_events(df_report, "cdm_"),
            "events_freq": aggregate_events(df_report, "freq_"),
            "events_rand": aggregate_events(df_report, "rand_"),
        },
        "last_n_window": {
            "hits": aggregate_hits(df_last_n),
            "events_cdm": aggregate_events(df_last_n, "cdm_"),
            "events_freq": aggregate_events(df_last_n, "freq_"),
            "events_rand": aggregate_events(df_last_n, "rand_"),
        }
    }

    save_json(SUMMARY_JSON, summary)

    print("Done.")
    print(f"- Predictions: {PREDICTIONS_CSV}")
    print(f"- Summary:     {SUMMARY_JSON}")
    print(f"- State:       {STATE_JSON}")


if __name__ == "__main__":
    main()
