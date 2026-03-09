#!/usr/bin/env python3
"""
Feature Selection Pipeline
===========================
After data imputation, select features using two methods:
  1. Multi-state Markov model HR values (R's msm package)  → table2.1
  2. Boruta algorithm (Random Forest-based)                → table2.2
  3. Cox-based feature selection on MSM set (Python):
     - LASSO-Cox
     - RFE-Cox

Steps:
  1. Directly load ana_dat_imputed.csv
  2. MSM HR-based feature selection via R script
     - Filter: HR < 0.90 or > 1.10  AND  P < 0.01
     - Output: analysis_set_msm.csv
  3. Boruta feature selection (D0 + D2)
     - D0: state_start=0 → predict state_stop ∈ {0,2,1}
     - D2: state_start=2 → predict state_stop ∈ {2,0,1}
     - Output: analysis_set_boruta.csv
  4. Cox feature selection on analysis_set_msm.csv (D0 + D2)
     - LASSO-Cox and RFE-Cox
     - Output: analysis_set_lasso_cox.csv, analysis_set_rfe_cox.csv

Usage:
  python feature_selection_pipeline.py \\
    --data-path out/feature_selection/ana_dat_imputed.csv \\
    --out-dir out/feature_selection

  # Skip MSM (already run) and only run Boruta:
  python feature_selection_pipeline.py --skip-msm

  # Skip Boruta and only run MSM:
  python feature_selection_pipeline.py --skip-boruta

  # Skip Cox selection:
  python feature_selection_pipeline.py --skip-cox
"""

from __future__ import annotations

import argparse
import subprocess
import time
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42

# Columns explicitly excluded from features (user specification)
EXCLUDE_COLS = {
    "id", "gryhb_c", "t_start", "t_stop",
    "wave_exam", "time", "birthday_updated",
}

# Event columns (define multi-state transitions, not features)
EVENT_COLS = {"state_start", "state_stop"}


# ──────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────
def _timer():
    return time.perf_counter()


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  → Saved {path}  ({df.shape[0]} rows × {df.shape[1]} cols)")


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns (all cols except excluded/event cols).

    User spec: features = all columns except
    {id, gryhb_c, t_start, t_stop, wave_exam, time, gender, ethnichan}.
    state_start / state_stop define multi-state events.
    """
    non_feature = EXCLUDE_COLS | EVENT_COLS
    return sorted(set(df.columns) - non_feature)


def load_imputed_data(data_path: Path) -> pd.DataFrame:
    """Load the imputed analysis dataset directly."""
    if not data_path.exists():
        raise FileNotFoundError(f"Required file not found: {data_path}")
    df = pd.read_csv(data_path)

    required_cols = {"id", "state_start", "state_stop"}
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(
            f"Input data missing required columns: {missing}"
        )
    return df


def _coerce_feature_for_boruta(s: pd.Series) -> pd.Series:
    """Convert feature series to numeric for Boruta, keeping NaN where possible."""
    if pd.api.types.is_bool_dtype(s):
        return s.astype(float)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")

    s_str = s.astype("string").str.strip()
    s_str = s_str.mask(s_str.isin(["", "NA", "NaN", "None", "null"]))

    s_num = pd.to_numeric(s_str, errors="coerce")
    if len(s_num) > 0 and s_num.notna().mean() >= 0.9:
        return s_num

    cat = s_str.astype("category")
    codes = cat.cat.codes.astype(float)
    codes[codes < 0] = np.nan
    return pd.Series(codes, index=s.index)


# ══════════════════════════════════════════════════════════════
# STEP 2  —  MSM feature selection (calls R script)
# ══════════════════════════════════════════════════════════════
def run_msm_feature_selection(
    data_path: Path,
    out_dir: Path,
    r_script: str = "feature_selection_msm.R",
) -> List[str]:
    """Run R's msm-based feature selection and return significant features."""
    print("\n" + "─" * 70)
    print("  STEP 2: MSM feature selection via R's msm package")
    print("─" * 70)

    r_script_path = Path(r_script).resolve()
    if not r_script_path.exists():
        raise FileNotFoundError(
            f"R script not found: {r_script_path}\n"
            "Please ensure feature_selection_msm.R is in the project root."
        )

    cmd = [
        "Rscript",
        str(r_script_path),
        str(data_path.resolve()),
        str(out_dir.resolve()),
    ]
    print(f"  Running: {' '.join(cmd)}")
    print(f"  (this may take a long time for many features)\n")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"\n  ⚠ R script exited with code {result.returncode}")

    # Read significant features
    sig_path = out_dir / "msm_significant_features.csv"
    if sig_path.exists():
        sig_df = pd.read_csv(sig_path)
        features = sorted(sig_df["feature"].tolist())
    else:
        print("  ⚠ No significant features file produced.")
        features = []

    return features


# ══════════════════════════════════════════════════════════════
# STEP 3  —  Boruta feature selection
# ══════════════════════════════════════════════════════════════
def run_boruta_selection(
    df: pd.DataFrame,
    feature_cols: List[str],
    out_dir: Path,
    seed: int = RANDOM_SEED,
    max_iter: int = 100,
    n_estimators: int = 500,
) -> List[str]:
    """Run Boruta feature selection on D0 and D2 risk sets.

    D0: state_start = 0  → predict state_stop ∈ {0, 2, 1} (three-class)
    D2: state_start = 2  → predict state_stop ∈ {2, 0, 1} (three-class)

    Returns the union of Boruta-confirmed features from both datasets.
    """
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    print("\n" + "─" * 70)
    print("  STEP 3: Boruta feature selection")
    print("─" * 70)

    if len(feature_cols) == 0:
        print("  ⚠ No input features for Boruta (MSM-selected set is empty).")
        return []

    all_selected: set = set()
    detail_rows: List[dict] = []

    for ds_name, ss_val in [("D0", 0), ("D2", 2)]:
        print(f"\n  ── {ds_name}  (state_start = {ss_val}) ──")

        sub = df[
            (df["state_start"] == ss_val) &
            (df["state_stop"].isin([0, 1, 2]))
        ].copy()
        print(f"    Samples: {len(sub)}")

        if len(sub) < 30:
            print(f"    ⚠ Too few samples ({len(sub)}), skipping {ds_name}.")
            continue

        y = sub["state_stop"].astype(int).values
        n_classes = len(np.unique(y))
        print(f"    Classes: {np.unique(y)} ({n_classes}-class)")

        if n_classes < 2:
            print(f"    ⚠ Only 1 class present, skipping {ds_name}.")
            continue

        # Prepare features: encode non-numeric to numeric, then fill NaN
        X = sub[feature_cols].copy()
        for c in feature_cols:
            X[c] = _coerce_feature_for_boruta(X[c])

        nan_cols = X.columns[X.isna().any()].tolist()
        if nan_cols:
            print(f"    Filling NaN in {len(nan_cols)} columns with median")
            for c in nan_cols:
                med = X[c].median()
                X[c] = X[c].fillna(med if pd.notna(med) else 0.0)

        X_arr = X.values.astype(np.float64)

        # Run Boruta
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced",
        )
        boruta = BorutaPy(
            rf,
            n_estimators="auto",
            random_state=seed,
            max_iter=max_iter,
            verbose=2,
        )

        print(f"    Running Boruta (max_iter={max_iter}) ...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            boruta.fit(X_arr, y)

        # Selected key features = Confirmed only
        confirmed = [
            feature_cols[i]
            for i in range(len(feature_cols))
            if boruta.support_[i]
        ]
        tentative = [
            feature_cols[i]
            for i in range(len(feature_cols))
            if boruta.support_weak_[i]
        ]
        selected = confirmed
        all_selected.update(selected)

        # Collect detail
        for i, col in enumerate(feature_cols):
            detail_rows.append({
                "dataset": ds_name,
                "feature": col,
                "ranking": int(boruta.ranking_[i]),
                "confirmed": bool(boruta.support_[i]),
                "tentative": bool(boruta.support_weak_[i]),
                "selected_confirmed": col in selected,
            })

        print(f"\n    {ds_name} results:")
        print(f"      Confirmed: {len(confirmed)}")
        print(f"      Tentative: {len(tentative)}")
        print(f"      Selected (confirmed): {len(selected)}")

        print(f"\n    {ds_name} selected features:")
        for f in sorted(selected):
            print(f"      ✓ {f}")

    # Save Boruta detail
    detail_df = pd.DataFrame(detail_rows)
    if len(detail_df) > 0:
        save_df(detail_df, out_dir / "boruta_detail.csv")

    final_features = sorted(all_selected)

    print(f"\n  ── Boruta union (D0 ∪ D2) ──")
    print(f"    Total unique features: {len(final_features)}")
    for f in final_features:
        print(f"      ✓ {f}")

    return final_features


# ══════════════════════════════════════════════════════════════
# STEP 4  —  Cox feature selection (LASSO-Cox + RFE-Cox)
# ══════════════════════════════════════════════════════════════
def _compute_time_years(df: pd.DataFrame) -> pd.Series:
    time_raw = pd.to_numeric(df.get("time"), errors="coerce")
    if "t_start" in df.columns and "t_stop" in df.columns:
        t_start = pd.to_datetime(df["t_start"], errors="coerce")
        t_stop = pd.to_datetime(df["t_stop"], errors="coerce")
        date_time = (t_stop - t_start).dt.days / 365.25
        bad = time_raw.isna() | (time_raw <= 0)
        time_raw = time_raw.where(~bad, date_time)
    return pd.to_numeric(time_raw, errors="coerce")


def _build_cox_cache(X: np.ndarray, time: np.ndarray, event: np.ndarray) -> dict:
    order = np.argsort(-time, kind="mergesort")
    Xs = X[order]
    ts = time[order]
    es = event[order].astype(int)

    risk_end = []
    d_list = []
    event_x = []
    i = 0
    n = len(ts)
    while i < n:
        j = i
        while j + 1 < n and ts[j + 1] == ts[i]:
            j += 1
        e_block = es[i:j + 1]
        d = int(e_block.sum())
        if d > 0:
            risk_end.append(j)
            d_list.append(d)
            event_x.append(Xs[i:j + 1][e_block == 1].sum(axis=0))
        i = j + 1

    if len(risk_end) == 0:
        raise ValueError("No events available for Cox model.")

    return {
        "X": Xs.astype(np.float64, copy=False),
        "risk_end": np.asarray(risk_end, dtype=int),
        "d": np.asarray(d_list, dtype=np.float64),
        "event_x": np.vstack(event_x).astype(np.float64, copy=False),
        "n": Xs.shape[0],
    }


def _cox_nll_and_grad(beta: np.ndarray, cache: dict, l2: float = 0.0) -> tuple[float, np.ndarray]:
    X = cache["X"]
    risk_end = cache["risk_end"]
    d = cache["d"]
    event_x = cache["event_x"]
    n = cache["n"]

    eta = np.clip(X @ beta, -50.0, 50.0)
    exp_eta = np.exp(eta)
    cum_exp = np.cumsum(exp_eta)
    cum_xexp = np.cumsum(X * exp_eta[:, None], axis=0)

    risk_sum = np.maximum(cum_exp[risk_end], 1e-12)
    risk_x = cum_xexp[risk_end]

    loglik = float(np.sum(event_x @ beta) - np.sum(d * np.log(risk_sum)))
    grad = event_x.sum(axis=0) - ((d / risk_sum)[:, None] * risk_x).sum(axis=0)

    if l2 > 0:
        loglik -= 0.5 * l2 * float(np.dot(beta, beta))
        grad -= l2 * beta

    nll = -loglik / n
    g = -grad / n
    return nll, g


def _cox_nll_only(beta: np.ndarray, cache: dict) -> float:
    return _cox_nll_and_grad(beta, cache, l2=0.0)[0]


def _soft_threshold(x: np.ndarray, lam: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - lam, 0.0)


def _fit_cox_l2(X: np.ndarray, time: np.ndarray, event: np.ndarray, l2: float = 1e-4) -> tuple[np.ndarray, float]:
    cache = _build_cox_cache(X, time, event)
    p = X.shape[1]

    def fun(b):
        return _cox_nll_and_grad(b, cache, l2=l2)[0]

    def jac(b):
        return _cox_nll_and_grad(b, cache, l2=l2)[1]

    res = minimize(
        fun=fun,
        x0=np.zeros(p, dtype=float),
        jac=jac,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-10},
    )
    beta = res.x if res.x is not None else np.zeros(p, dtype=float)
    nll = float(fun(beta))
    return beta, nll


def _fit_lasso_cox_fista(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    alpha: float,
    max_iter: int = 300,
    tol: float = 1e-6,
) -> np.ndarray:
    cache = _build_cox_cache(X, time, event)
    p = X.shape[1]
    beta = np.zeros(p, dtype=float)
    y = beta.copy()
    t = 1.0
    L = 1.0
    obj_prev = np.inf

    for _ in range(max_iter):
        nll_y, grad_y = _cox_nll_and_grad(y, cache, l2=0.0)
        while True:
            b_new = _soft_threshold(y - grad_y / L, alpha / L)
            nll_new = _cox_nll_only(b_new, cache)
            diff = b_new - y
            q = nll_y + np.dot(grad_y, diff) + 0.5 * L * np.dot(diff, diff)
            if nll_new <= q + 1e-10:
                break
            L *= 2.0
            if L > 1e7:
                break

        obj = nll_new + alpha * np.sum(np.abs(b_new))
        if abs(obj_prev - obj) < tol:
            beta = b_new
            break
        obj_prev = obj

        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        y = b_new + ((t - 1) / t_new) * (b_new - beta)
        beta = b_new
        t = t_new

    return beta


def _prepare_cox_input(
    df: pd.DataFrame,
    feature_cols: List[str],
    seed: int,
    max_rows: int = 12000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    sub = df.copy()
    for c in feature_cols:
        sub[c] = _coerce_feature_for_boruta(sub[c])
        med = sub[c].median(skipna=True)
        sub[c] = sub[c].fillna(med if pd.notna(med) else 0.0)

    keep_cols = []
    for c in feature_cols:
        v = pd.to_numeric(sub[c], errors="coerce")
        if v.notna().sum() < 20:
            continue
        if float(v.std(ddof=0)) <= 1e-12:
            continue
        keep_cols.append(c)
    if len(keep_cols) == 0:
        return np.empty((0, 0)), np.empty(0), np.empty(0), []

    X = sub[keep_cols].to_numpy(dtype=np.float64)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd = np.where(sd <= 1e-12, 1.0, sd)
    X = (X - mu) / sd

    t = pd.to_numeric(sub["time_years"], errors="coerce").to_numpy(dtype=np.float64)
    e = pd.to_numeric(sub["event"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)

    valid = np.isfinite(t) & (t > 0) & np.isfinite(X).all(axis=1)
    X = X[valid]
    t = t[valid]
    e = e[valid]
    if X.shape[0] == 0:
        return np.empty((0, 0)), np.empty(0), np.empty(0), []

    if X.shape[0] > max_rows:
        rng = np.random.default_rng(seed)
        idx_event = np.where(e == 1)[0]
        idx_nonevent = np.where(e == 0)[0]
        n_event = min(len(idx_event), max(1, int(max_rows * len(idx_event) / len(e))))
        n_nonevent = max_rows - n_event
        pick_e = rng.choice(idx_event, size=n_event, replace=False) if n_event < len(idx_event) else idx_event
        pick_n = rng.choice(idx_nonevent, size=min(n_nonevent, len(idx_nonevent)), replace=False)
        keep = np.concatenate([pick_e, pick_n])
        rng.shuffle(keep)
        X = X[keep]
        t = t[keep]
        e = e[keep]

    return X, t, e, keep_cols


def _select_lasso_cox_features(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    feature_names: List[str],
    seed: int,
) -> List[str]:
    if X.shape[0] < 100 or X.shape[1] == 0 or np.unique(e).size < 2:
        return []

    alphas = np.logspace(-3, -0.5, 7)
    kf = KFold(n_splits=3, shuffle=True, random_state=seed)
    best_alpha = alphas[0]
    best_score = np.inf

    for a in alphas:
        fold_scores = []
        for tr, va in kf.split(X):
            if np.unique(e[tr]).size < 2 or np.unique(e[va]).size < 2:
                continue
            try:
                beta = _fit_lasso_cox_fista(X[tr], t[tr], e[tr], alpha=float(a))
                score = _cox_nll_only(beta, _build_cox_cache(X[va], t[va], e[va]))
                fold_scores.append(score)
            except Exception:
                continue
        if len(fold_scores) == 0:
            continue
        m = float(np.mean(fold_scores))
        if m < best_score:
            best_score = m
            best_alpha = a

    beta = _fit_lasso_cox_fista(X, t, e, alpha=float(best_alpha))
    idx = np.where(np.abs(beta) > 1e-6)[0]
    if len(idx) == 0:
        idx = np.argsort(-np.abs(beta))[: max(1, min(5, len(beta)))]
    return sorted({feature_names[i] for i in idx})


def _select_rfe_cox_features(
    X: np.ndarray,
    t: np.ndarray,
    e: np.ndarray,
    feature_names: List[str],
) -> List[str]:
    p = X.shape[1]
    if X.shape[0] < 100 or p == 0 or np.unique(e).size < 2:
        return []

    active = list(range(p))
    min_keep = max(5, int(np.ceil(np.sqrt(p))))
    min_keep = min(min_keep, p)
    best_active = active.copy()
    best_aic = np.inf

    while len(active) >= min_keep:
        Xa = X[:, active]
        try:
            beta, nll = _fit_cox_l2(Xa, t, e, l2=1e-4)
        except Exception:
            break
        k = len(active)
        aic = 2 * k + 2 * (nll * Xa.shape[0])
        if np.isfinite(aic) and aic < best_aic:
            best_aic = float(aic)
            best_active = active.copy()

        if len(active) == min_keep:
            break
        rm_pos = int(np.argmin(np.abs(beta)))
        del active[rm_pos]
        if len(active) == 0:
            break

    return sorted({feature_names[i] for i in best_active})


def run_cox_feature_selection(
    msm_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    out_dir: Path,
    seed: int = RANDOM_SEED,
) -> None:
    print("\n" + "─" * 70)
    print("  STEP 4: Cox feature selection (LASSO-Cox + RFE-Cox)")
    print("─" * 70)

    if len(msm_df) == 0:
        print("  ⚠ MSM analysis set is empty; skip Cox selection.")
        return

    feature_cols = [c for c in msm_df.columns if c not in {"id", "state_start", "state_stop"}]
    if len(feature_cols) == 0:
        print("  ⚠ No MSM-selected features for Cox selection.")
        return

    time_years = _compute_time_years(raw_df)
    cox_df = msm_df.copy().reset_index(drop=True)
    cox_df["time_years"] = pd.to_numeric(time_years.reset_index(drop=True), errors="coerce")
    cox_df["event"] = (pd.to_numeric(cox_df["state_stop"], errors="coerce") == 1).astype(int)

    by_ds = {}
    for ds_name, ss_val in [("D0", 0), ("D2", 2)]:
        sub = cox_df[
            (pd.to_numeric(cox_df["state_start"], errors="coerce") == ss_val) &
            (pd.to_numeric(cox_df["state_stop"], errors="coerce").isin([0, 1, 2])) &
            cox_df["time_years"].notna() &
            (cox_df["time_years"] > 0)
        ].copy()

        X, t, e, usable_feats = _prepare_cox_input(
            sub, feature_cols=feature_cols, seed=seed + ss_val, max_rows=12000
        )
        if X.shape[0] == 0 or np.unique(e).size < 2:
            print(f"\n  {ds_name}: insufficient valid samples/events for Cox; skip.")
            by_ds[ds_name] = {"lasso": [], "rfe": []}
            continue

        lasso_feats = _select_lasso_cox_features(X, t, e, usable_feats, seed=seed + ss_val)
        rfe_feats = _select_rfe_cox_features(X, t, e, usable_feats)
        by_ds[ds_name] = {"lasso": lasso_feats, "rfe": rfe_feats}

        print(f"\n  {ds_name} LASSO-Cox selected ({len(lasso_feats)}):")
        for f in lasso_feats:
            print(f"    ✓ {f}")
        print(f"\n  {ds_name} RFE-Cox selected ({len(rfe_feats)}):")
        for f in rfe_feats:
            print(f"    ✓ {f}")

    lasso_rows = []
    rfe_rows = []
    for ds_name in ["D0", "D2"]:
        for f in by_ds.get(ds_name, {}).get("lasso", []):
            lasso_rows.append({"dataset": ds_name, "feature": f})
        for f in by_ds.get(ds_name, {}).get("rfe", []):
            rfe_rows.append({"dataset": ds_name, "feature": f})

    lasso_df = pd.DataFrame(lasso_rows)
    rfe_df = pd.DataFrame(rfe_rows)
    save_df(lasso_df, out_dir / "lasso_cox_features_by_dataset.csv")
    save_df(rfe_df, out_dir / "rfe_cox_features_by_dataset.csv")

    def build_set(base: pd.DataFrame, feats: List[str]) -> pd.DataFrame:
        feats_keep = [f for f in feats if f in base.columns]
        cols = ["id", "state_start", "state_stop"] + feats_keep
        return base[cols].copy()

    d0_base = msm_df[pd.to_numeric(msm_df["state_start"], errors="coerce") == 0].copy()
    d2_base = msm_df[pd.to_numeric(msm_df["state_start"], errors="coerce") == 2].copy()

    save_df(build_set(d0_base, by_ds.get("D0", {}).get("lasso", [])), out_dir / "analysis_set_lasso_cox_D0.csv")
    save_df(build_set(d2_base, by_ds.get("D2", {}).get("lasso", [])), out_dir / "analysis_set_lasso_cox_D2.csv")
    save_df(build_set(d0_base, by_ds.get("D0", {}).get("rfe", [])), out_dir / "analysis_set_rfe_cox_D0.csv")
    save_df(build_set(d2_base, by_ds.get("D2", {}).get("rfe", [])), out_dir / "analysis_set_rfe_cox_D2.csv")

    lasso_union = sorted(set(by_ds.get("D0", {}).get("lasso", [])) | set(by_ds.get("D2", {}).get("lasso", [])))
    rfe_union = sorted(set(by_ds.get("D0", {}).get("rfe", [])) | set(by_ds.get("D2", {}).get("rfe", [])))
    save_df(build_set(msm_df, lasso_union), out_dir / "analysis_set_lasso_cox.csv")
    save_df(build_set(msm_df, rfe_union), out_dir / "analysis_set_rfe_cox.csv")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Feature Selection Pipeline (MSM + Boruta)"
    )
    p.add_argument("--data-path", default="out/feature_selection/ana_dat_imputed.csv",
                   help="Path to imputed analysis CSV (ana_dat_imputed.csv)")
    p.add_argument("--out-dir", default="out/feature_selection",
                   help="Output directory")
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--skip-msm", action="store_true",
                   help="Skip MSM step (use existing results if available)")
    p.add_argument("--skip-boruta", action="store_true",
                   help="Skip Boruta step (use existing results if available)")
    p.add_argument("--skip-cox", action="store_true",
                   help="Skip Cox feature selection step")
    p.add_argument("--boruta-max-iter", type=int, default=100,
                   help="Boruta max iterations")
    p.add_argument("--boruta-n-estimators", type=int, default=500,
                   help="Random Forest n_estimators for Boruta")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed
    t0 = _timer()

    print("=" * 70)
    print("  Feature Selection Pipeline")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════════
    #  STEP 1  —  Load ana_dat_imputed.csv directly
    # ═══════════════════════════════════════════════════════════
    print("\n" + "─" * 70)
    print("  STEP 1: Load imputed dataset (ana_dat_imputed.csv)")
    print("─" * 70)

    print(f"  Input: {data_path}")
    merged = load_imputed_data(data_path)
    merged["state_start"] = pd.to_numeric(merged["state_start"], errors="coerce")
    merged["state_stop"] = pd.to_numeric(merged["state_stop"], errors="coerce")

    feature_cols = get_feature_cols(merged)
    print(f"  Total rows: {len(merged)}")
    print(f"  Unique IDs: {merged['id'].nunique()}")
    print(f"  Feature columns: {len(feature_cols)}")
    if len(feature_cols) != 91:
        print(f"  ⚠ Expected 91 features by definition, found {len(feature_cols)}")
    print(f"  Features: {feature_cols}")

    # ═══════════════════════════════════════════════════════════
    #  STEP 2  —  MSM feature selection via R
    # ═══════════════════════════════════════════════════════════
    if not args.skip_msm:
        msm_features = run_msm_feature_selection(
            data_path=data_path,
            out_dir=out_dir,
            r_script="feature_selection_msm.R",
        )
    else:
        sig_path = out_dir / "msm_significant_features.csv"
        if sig_path.exists():
            msm_features = sorted(
                pd.read_csv(sig_path)["feature"].tolist()
            )
            print(f"\n  [skip-msm] Loaded {len(msm_features)} MSM features "
                  f"from {sig_path}")
        else:
            msm_features = []
            print(f"\n  [skip-msm] No existing MSM results found.")

    print(f"\n  MSM significant features: {len(msm_features)}")

    # Save MSM analysis set
    if msm_features:
        msm_keep = [f for f in msm_features if f in merged.columns]
        msm_cols = ["id", "state_start", "state_stop"] + msm_keep
        msm_analysis = merged[msm_cols].copy()
        save_df(msm_analysis, out_dir / "analysis_set_msm.csv")
    else:
        print("  ⚠ No MSM features selected; analysis_set_msm.csv not saved.")

    # ═══════════════════════════════════════════════════════════
    #  STEP 3  —  Boruta feature selection
    # ═══════════════════════════════════════════════════════════
    boruta_input_features = [f for f in msm_features if f in merged.columns]
    print(f"\n  Boruta input features (from MSM): {len(boruta_input_features)}")
    if boruta_input_features:
        print(f"  Boruta input list: {sorted(boruta_input_features)}")

    if not args.skip_boruta:
        boruta_features = run_boruta_selection(
            df=merged,
            feature_cols=boruta_input_features,
            out_dir=out_dir,
            seed=seed,
            max_iter=args.boruta_max_iter,
            n_estimators=args.boruta_n_estimators,
        )
    else:
        boruta_path = out_dir / "boruta_selected_features.csv"
        if boruta_path.exists():
            boruta_features = sorted(
                pd.read_csv(boruta_path)["feature"].tolist()
            )
            print(f"\n  [skip-boruta] Loaded {len(boruta_features)} Boruta "
                  f"features from {boruta_path}")
        else:
            boruta_features = []
            print(f"\n  [skip-boruta] No existing Boruta results found.")

    if boruta_features:
        boruta_features = sorted(set(boruta_features) & set(boruta_input_features))

    print(f"\n  Boruta selected features: {len(boruta_features)}")

    # Save Boruta features list
    boruta_feat_df = pd.DataFrame({"feature": boruta_features})
    save_df(boruta_feat_df, out_dir / "boruta_selected_features.csv")

    # Save Boruta analysis set
    if boruta_features:
        boruta_keep = [f for f in boruta_features if f in merged.columns]
        boruta_cols = ["id", "state_start", "state_stop"] + boruta_keep
        boruta_analysis = merged[boruta_cols].copy()
        save_df(boruta_analysis, out_dir / "analysis_set_boruta.csv")
    else:
        print("  ⚠ No Boruta features selected; "
              "analysis_set_boruta.csv not saved.")

    # ═══════════════════════════════════════════════════════════
    #  STEP 4  —  Cox feature selection on MSM analysis set
    # ═══════════════════════════════════════════════════════════
    if not args.skip_cox:
        if msm_features:
            run_cox_feature_selection(
                msm_df=msm_analysis,
                raw_df=merged,
                out_dir=out_dir,
                seed=seed,
            )
        else:
            print("\n  ⚠ No MSM features selected; skip Cox feature selection.")
    else:
        print("\n  [skip-cox] Skip Cox feature selection.")

    # ═══════════════════════════════════════════════════════════
    #  Summary
    # ═══════════════════════════════════════════════════════════
    elapsed = round(_timer() - t0, 1)

    print("\n" + "=" * 70)
    print("  Feature Selection Summary")
    print("=" * 70)
    print(f"  MSM features:    {len(msm_features)}")
    print(f"  Boruta features: {len(boruta_features)}")

    if msm_features and boruta_features:
        overlap = sorted(set(msm_features) & set(boruta_features))
        union = sorted(set(msm_features) | set(boruta_features))
        print(f"  Overlap:         {len(overlap)}")
        print(f"  Union:           {len(union)}")
        if overlap:
            print(f"\n  Overlapping features:")
            for f in overlap:
                print(f"    ✓ {f}")

    print(f"\n  Elapsed: {elapsed}s")
    print(f"  All outputs → {out_dir}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
