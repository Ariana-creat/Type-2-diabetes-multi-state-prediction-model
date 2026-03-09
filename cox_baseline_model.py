#!/usr/bin/env python3
"""
cox_baseline_model.py
Baseline Cox Proportional Hazards Models for T2D Risk Prediction.

Competing risks framework:
  - Cause 1 (State 1): Type 2 Diabetes (absorbing)
  - Cause 2 (State 2): Pre-diabetes / Impaired glucose regulation

Models 1–4 use progressively richer feature sets.
Pipeline: FP transformation → VIF check → Cox fit → PH test → Age interactions
          → Recalibrate on validation → CIF (Aalen–Johansen) → Evaluate on test
"""

import os
import sys
import warnings
import itertools
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test
from lifelines.utils import concordance_index
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from scipy.stats import chi2, norm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "PingFang SC", "Heiti TC", "STHeiti", "SimHei", "DejaVu Sans"
]
matplotlib.rcParams["axes.unicode_minus"] = False

# ====================================================================
# Configuration
# ====================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMPUTED_PATH = os.path.join(BASE_DIR, "out", "feature_selection", "ana_dat_imputed.csv")
SPLIT_PATH = os.path.join(BASE_DIR, "out", "preprocess", "id_split_map.csv")
OUT_DIR = os.path.join(BASE_DIR, "out", "cox_baseline", "cox_baseline_fix")
SEED = 42
np.random.seed(SEED)

MODEL_FEATURES = {
    "Model1": [
        "age", "CVD", "ethnichan", "fam_diabetes", "hyperlipid",
        "hypertension", "lifestyle_score", "shift_work", "weight",
    ],
    "Model2": [
        "age", "CVD", "ethnichan", "fam_diabetes", "hyperlipid",
        "hypertension", "lifestyle_score", "shift_work", "weight",
        "DBP", "HR", "SBP",
    ],
    "Model3": [
        "age", "CVD", "ethnichan", "fam_diabetes", "hyperlipid",
        "hypertension", "lifestyle_score", "shift_work", "weight",
        "DBP", "HR", "SBP", "FPG",
    ],
    "Model4": [
        "age", "CVD", "ethnichan", "fam_diabetes", "hyperlipid",
        "hypertension", "lifestyle_score", "shift_work", "weight",
        "DBP", "HR", "SBP", "HbA1c",
    ],
}

CONTINUOUS_VARS = {"age", "weight", "lifestyle_score", "DBP", "HR", "SBP", "FPG", "HbA1c"}
CATEGORICAL_VARS = {"CVD", "ethnichan", "fam_diabetes", "hyperlipid", "hypertension", "shift_work"}
FP_POWERS = [-2, -1, -0.5, 0, 0.5, 1, 2, 3]
HORIZONS = {1: 10, 2: 5}


# ====================================================================
# Fractional Polynomial helpers
# ====================================================================
def _fp_col(x, power, shift):
    xs = np.maximum(x + shift, 1e-6)
    return np.log(xs) if power == 0 else np.power(xs, power)


def _fp2_cols(x, p1, p2, shift):
    xs = np.maximum(x + shift, 1e-6)
    t1 = _fp_col(x, p1, shift)
    t2 = t1 * np.log(xs) if p1 == p2 else _fp_col(x, p2, shift)
    return t1, t2


def _standardize(arr, mean=None, std=None):
    if mean is None:
        mean = np.nanmean(arr)
    if std is None:
        std = np.nanstd(arr) + 1e-8
    return (arr - mean) / std, mean, std


def select_fp_for_variable(train_df, var, other_features, dur_col, evt_col):
    """Select best FP type for *var* by LRT within the full model."""
    x = train_df[var].values.astype(float)
    shift = max(0.0, -np.nanmin(x) + 1.0)
    base_cols = [c for c in other_features if c != var and c in train_df.columns]
    base = train_df[base_cols + [dur_col, evt_col]].copy()

    def _fit_ll(extra_df):
        df = pd.concat([base, extra_df], axis=1).dropna()
        if len(df) < 50:
            return -np.inf
        try:
            cph = CoxPHFitter(penalizer=0.001)
            cph.fit(df, duration_col=dur_col, event_col=evt_col, show_progress=False)
            return cph.log_likelihood_
        except Exception:
            return -np.inf

    ll_lin = _fit_ll(pd.DataFrame({var: x}))

    best_fp1_ll, best_fp1_p = -np.inf, 1
    for p in FP_POWERS:
        if p == 1:
            continue
        t = _fp_col(x, p, shift)
        ts, _, _ = _standardize(t)
        ll = _fit_ll(pd.DataFrame({f"{var}_fp": ts}))
        if ll > best_fp1_ll:
            best_fp1_ll, best_fp1_p = ll, p

    best_fp2_ll, best_fp2_ps = -np.inf, (1, 1)
    for i, p1 in enumerate(FP_POWERS):
        for p2 in FP_POWERS[i:]:
            t1, t2 = _fp2_cols(x, p1, p2, shift)
            t1s, _, _ = _standardize(t1)
            t2s, _, _ = _standardize(t2)
            ll = _fit_ll(pd.DataFrame({f"{var}_fp1": t1s, f"{var}_fp2": t2s}))
            if ll > best_fp2_ll:
                best_fp2_ll, best_fp2_ps = ll, (p1, p2)

    alpha = 0.05
    dev_fp2 = 2 * (best_fp2_ll - ll_lin) if np.isfinite(best_fp2_ll) else 0
    dev_fp1 = 2 * (best_fp1_ll - ll_lin) if np.isfinite(best_fp1_ll) else 0
    dev_21 = 2 * (best_fp2_ll - best_fp1_ll) if np.isfinite(best_fp2_ll) and np.isfinite(best_fp1_ll) else 0

    if dev_fp2 > 0 and 1 - chi2.cdf(dev_fp2, 3) < alpha:
        if dev_21 > 0 and 1 - chi2.cdf(dev_21, 2) < alpha:
            return {"type": "fp2", "powers": list(best_fp2_ps), "shift": shift}
        if dev_fp1 > 0 and 1 - chi2.cdf(dev_fp1, 1) < alpha:
            return {"type": "fp1", "powers": [best_fp1_p], "shift": shift}
    elif dev_fp1 > 0 and 1 - chi2.cdf(dev_fp1, 1) < alpha:
        return {"type": "fp1", "powers": [best_fp1_p], "shift": shift}
    return {"type": "linear", "powers": [1], "shift": shift}


def apply_fp_transforms(df, cont_features, cat_features, fp_specs, fit_stats=None):
    """Apply FP transformations. Returns (result_df, feature_list, stats)."""
    result = df.copy()
    feature_list = list(cat_features)
    stats = {} if fit_stats is None else dict(fit_stats)
    fp_groups = {c: [c] for c in cat_features}

    for var in cont_features:
        spec = fp_specs.get(var)
        if spec is None or spec["type"] == "linear":
            feature_list.append(var)
            fp_groups[var] = [var]
            continue

        x = result[var].values.astype(float)
        shift = spec["shift"]

        if spec["type"] == "fp1":
            p = spec["powers"][0]
            t = _fp_col(x, p, shift)
            col = f"{var}_fp{p}"
            if fit_stats is None:
                ts, m, s = _standardize(t)
                stats[col] = {"mean": m, "std": s}
            else:
                ts = (t - stats[col]["mean"]) / stats[col]["std"]
            result[col] = ts
            feature_list.append(col)
            fp_groups[var] = [col]

        elif spec["type"] == "fp2":
            p1, p2 = spec["powers"]
            t1, t2 = _fp2_cols(x, p1, p2, shift)
            col1 = f"{var}_fp{p1}"
            col2 = f"{var}_fp{p2}b" if p1 == p2 else f"{var}_fp{p2}"
            if fit_stats is None:
                t1s, m1, s1 = _standardize(t1)
                t2s, m2, s2 = _standardize(t2)
                stats[col1] = {"mean": m1, "std": s1}
                stats[col2] = {"mean": m2, "std": s2}
            else:
                t1s = (t1 - stats[col1]["mean"]) / stats[col1]["std"]
                t2s = (t2 - stats[col2]["mean"]) / stats[col2]["std"]
            result[col1] = t1s
            result[col2] = t2s
            feature_list.append(col1)
            feature_list.append(col2)
            fp_groups[var] = [col1, col2]

    return result, feature_list, stats, fp_groups


# ====================================================================
# VIF check
# ====================================================================
def check_vif(df, features, fp_groups=None, threshold=5.0, prefer_interactions=False):
    """Iteratively remove features with VIF > threshold."""
    current = list(features)
    fp_groups = fp_groups or {f: [f] for f in current}
    removed = []
    while len(current) >= 2:
        X = df[current].apply(pd.to_numeric, errors="coerce").fillna(0).values
        Xc = add_constant(X)
        vifs = np.array([variance_inflation_factor(Xc, i + 1) for i in range(X.shape[1])])
        over_idx = np.where((~np.isfinite(vifs)) | (vifs > threshold))[0]
        if len(over_idx) == 0:
            break
        choose_idx = over_idx
        if prefer_interactions:
            interaction_idx = [
                i for i in over_idx
                if current[i].startswith("age_x_") or current[i].endswith("_x_logt")
            ]
            if interaction_idx:
                choose_idx = np.array(interaction_idx, dtype=int)
        max_idx = choose_idx[np.argmax(vifs[choose_idx])]
        worst = current[max_idx]
        print(f"    VIF: removing {worst} (VIF={vifs[max_idx]:.1f})")
        orig_var = None
        for ov, cols in fp_groups.items():
            if worst in cols:
                orig_var = ov
                break
        if orig_var and len(fp_groups.get(orig_var, [])) > 1:
            for col in fp_groups[orig_var]:
                if col in current:
                    current.remove(col)
                    removed.append({
                        "feature": col,
                        "original_variable": orig_var,
                        "vif": float(vifs[max_idx]),
                    })
        else:
            current.remove(worst)
            removed.append({
                "feature": worst,
                "original_variable": orig_var or worst,
                "vif": float(vifs[max_idx]),
            })
    return current, removed


# ====================================================================
# Breslow baseline hazard & censoring KM
# ====================================================================
def breslow_baseline(times, events, lp):
    rr = np.exp(np.clip(lp, -20, 20))
    ev_t = np.sort(np.unique(times[events == 1]))
    recs = []
    for t in ev_t:
        d = np.sum((np.abs(times - t) < 1e-10) & (events == 1))
        rs = np.sum(rr[times >= t - 1e-10])
        recs.append({"time": float(t), "dh": float(d / max(rs, 1e-10))})
    return pd.DataFrame(recs) if recs else pd.DataFrame({"time": pd.Series(dtype=float), "dh": pd.Series(dtype=float)})


def make_G_func(train_times, train_statuses):
    """KM estimate of censoring survival G(t) = P(C > t)."""
    censor_evt = (train_statuses == 0).astype(int)
    kmf = KaplanMeierFitter()
    kmf.fit(train_times, event_observed=censor_evt)
    km_t = np.asarray(kmf.timeline)
    km_s = np.asarray(kmf.survival_function_).flatten()

    def G(tq):
        tq = np.atleast_1d(np.asarray(tq, dtype=float))
        out = np.ones(len(tq))
        for i, ti in enumerate(tq):
            idx = np.where(km_t <= ti)[0]
            if len(idx):
                out[i] = km_s[idx[-1]]
        return np.maximum(out, 1e-6) if len(out) > 1 else max(float(out[0]), 1e-6)
    return G


# ====================================================================
# CIF Computation (Aalen-Johansen via cause-specific hazards)
# ====================================================================
def compute_cif(lp1, lp2, base1, base2, time_int_X1=None, time_int_beta1=None,
                time_int_X2=None, time_int_beta2=None, max_time=None):
    """Compute CIF using cause-specific hazard approach."""
    t1 = base1["time"].values if len(base1) else np.array([])
    t2 = base2["time"].values if len(base2) else np.array([])
    all_t = np.sort(np.unique(np.concatenate([t1, t2])))
    if max_time is not None:
        all_t = all_t[all_t <= max_time]
    if len(all_t) == 0:
        n = len(lp1)
        return {"times": np.array([]), "cif1": np.zeros((n, 0)), "cif2": np.zeros((n, 0))}

    dh1 = np.zeros(len(all_t))
    dh2 = np.zeros(len(all_t))
    for i, t in enumerate(all_t):
        m1 = np.abs(t1 - t) < 1e-10 if len(t1) else np.array([], dtype=bool)
        if m1.any():
            dh1[i] = base1.loc[m1, "dh"].values[0]
        m2 = np.abs(t2 - t) < 1e-10 if len(t2) else np.array([], dtype=bool)
        if m2.any():
            dh2[i] = base2.loc[m2, "dh"].values[0]

    n = len(lp1)
    S = np.ones(n)
    cif1 = np.zeros(n)
    cif2 = np.zeros(n)
    c1_mat = np.zeros((n, len(all_t)))
    c2_mat = np.zeros((n, len(all_t)))

    for k in range(len(all_t)):
        tk = all_t[k]
        log_tk = np.log(max(tk, 0.01))
        eff_lp1 = lp1.copy()
        eff_lp2 = lp2.copy()
        if time_int_X1 is not None and time_int_beta1 is not None:
            eff_lp1 += time_int_X1 @ time_int_beta1 * log_tk
        if time_int_X2 is not None and time_int_beta2 is not None:
            eff_lp2 += time_int_X2 @ time_int_beta2 * log_tk

        rr1 = np.exp(np.clip(eff_lp1, -20, 20))
        rr2 = np.exp(np.clip(eff_lp2, -20, 20))
        d1 = dh1[k] * rr1
        d2 = dh2[k] * rr2
        d_all = d1 + d2
        p_any = 1 - np.exp(-d_all)
        mask = d_all > 0
        p1 = np.where(mask, p_any * d1 / d_all, 0)
        p2 = np.where(mask, p_any * d2 / d_all, 0)
        cif1 += S * p1
        cif2 += S * p2
        S *= np.exp(-d_all)
        c1_mat[:, k] = np.clip(cif1, 0, 1)
        c2_mat[:, k] = np.clip(cif2, 0, 1)

    return {"times": all_t, "cif1": c1_mat, "cif2": c2_mat}


def interpolate_cif_at_horizon(cif_res, cause, horizon):
    """Extract risk at a specific horizon from CIF curves."""
    mat = cif_res["cif1"] if cause == 1 else cif_res["cif2"]
    tt = cif_res["times"]
    if len(tt) == 0:
        return np.zeros(mat.shape[0])
    idx = np.searchsorted(tt, horizon)
    if idx >= len(tt):
        return mat[:, -1]
    if idx == 0:
        return mat[:, 0]
    frac = (horizon - tt[idx - 1]) / max(tt[idx] - tt[idx - 1], 1e-10)
    return np.clip(mat[:, idx - 1] + frac * (mat[:, idx] - mat[:, idx - 1]), 0, 1)


# ====================================================================
# Observed CIF (Aalen-Johansen)
# ====================================================================
def aj_cif(times, statuses, cause, horizon):
    ok = np.isfinite(times) & np.isfinite(statuses)
    t, s = times[ok], statuses[ok].astype(int)
    ev_t = np.sort(np.unique(t[(t <= horizon) & (s != 0)]))
    S, cif = 1.0, 0.0
    for tt in ev_t:
        Y = np.sum(t >= tt)
        if Y <= 0:
            continue
        d_all = np.sum((np.abs(t - tt) < 1e-10) & (s != 0))
        d_c = np.sum((np.abs(t - tt) < 1e-10) & (s == cause))
        cif += S * (d_c / Y)
        S *= 1 - d_all / Y
    return np.clip(cif, 0, 1)


# ====================================================================
# IPCW Metrics
# ====================================================================
def ipcw_brier(times, statuses, risks, cause, horizon, G_func):
    ok = np.isfinite(times) & np.isfinite(statuses) & np.isfinite(risks)
    t, s, p = times[ok], statuses[ok].astype(int), np.clip(risks[ok], 1e-8, 1 - 1e-8)
    y = ((t <= horizon) & (s == cause)).astype(float)
    keep = ((t <= horizon) & (s != 0)) | (t > horizon)
    if keep.sum() < 20:
        return np.nan
    t, s, p, y = t[keep], s[keep], p[keep], y[keep]
    w = np.zeros(len(t))
    c1 = (t <= horizon) & (s != 0)
    c2 = t > horizon
    if c1.any():
        w[c1] = 1.0 / np.atleast_1d(G_func(t[c1] - 1e-8))
    if c2.any():
        gh = G_func(horizon)
        w[c2] = 1.0 / (gh if np.isscalar(gh) else gh[0])
    w[~np.isfinite(w)] = 0
    return float(np.sum(w * (y - p) ** 2) / max(np.sum(w), 1e-10)) if np.sum(w) > 0 else np.nan


def integrated_brier_score(times, statuses, cif_times, cif_matrix, cause, t_max, G_func):
    use_t = cif_times[(cif_times > 0) & (cif_times <= t_max)]
    if len(use_t) < 4:
        return np.nan
    if len(use_t) > 80:
        idx = np.unique(np.round(np.linspace(0, len(use_t) - 1, 80)).astype(int))
        use_t = use_t[idx]
    bs = []
    for h in use_t:
        ci = np.searchsorted(cif_times, h)
        ci = min(ci, cif_matrix.shape[1] - 1)
        risk = cif_matrix[:, ci] if ci == 0 else (
            cif_matrix[:, ci - 1] + (h - cif_times[ci - 1]) / max(cif_times[ci] - cif_times[ci - 1], 1e-10)
            * (cif_matrix[:, ci] - cif_matrix[:, ci - 1])
        )
        bs.append(ipcw_brier(times, statuses, np.clip(risk, 0, 1), cause, h, G_func))
    bs = np.array(bs)
    ok = np.isfinite(bs)
    if ok.sum() < 4:
        return np.nan
    x, y = use_t[ok], bs[ok]
    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2) / (x[-1] - x[0]))


def td_auc(times, statuses, risks, cause, horizon):
    """Time-dependent AUC for a specific cause at a horizon."""
    ok = np.isfinite(times) & np.isfinite(statuses) & np.isfinite(risks)
    t, s, r = times[ok], statuses[ok].astype(int), risks[ok]
    y = ((t <= horizon) & (s == cause)).astype(int)
    keep = ((t <= horizon) & (s != 0)) | (t > horizon)
    if keep.sum() < 30:
        return np.nan
    y_k, r_k = y[keep], r[keep]
    if y_k.sum() < 5 or (1 - y_k).sum() < 5:
        return np.nan
    try:
        return float(roc_auc_score(y_k, r_k))
    except Exception:
        return np.nan


def horizon_logistic_calibration_slope(times, statuses, risks, cause, horizon):
    """Calibration slope using horizon-specific predicted risk."""
    ok = np.isfinite(times) & np.isfinite(statuses) & np.isfinite(risks)
    t, s, r = times[ok], statuses[ok].astype(int), np.asarray(risks[ok], dtype=float)
    y = ((t <= horizon) & (s == cause)).astype(int)
    keep = ((t <= horizon) & (s != 0)) | (t > horizon)
    if keep.sum() < 30:
        return np.nan
    y_k = y[keep]
    if y_k.sum() < 5 or (1 - y_k).sum() < 5:
        return np.nan
    logit_r = np.log(np.clip(r[keep], 1e-6, 1 - 1e-6) / np.clip(1 - r[keep], 1e-6, 1 - 1e-6))
    try:
        lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        lr.fit(logit_r.reshape(-1, 1), y_k)
        return float(lr.coef_[0][0])
    except Exception:
        return np.nan


def royston_d(times, events, lp):
    ok = np.isfinite(times) & np.isfinite(events) & np.isfinite(lp)
    t, e, l = times[ok], events[ok].astype(int), lp[ok]
    if len(t) < 30 or e.sum() < 10:
        return np.nan, np.nan
    ranks = np.argsort(np.argsort(l)) + 1
    z = norm.ppf(ranks / (len(l) + 1))
    zdf = pd.DataFrame({"dur": t, "evt": e, "z": z})
    try:
        cph = CoxPHFitter(penalizer=0.0)
        cph.fit(zdf, duration_col="dur", event_col="evt", show_progress=False)
        beta = cph.params_["z"]
    except Exception:
        return np.nan, np.nan
    D = beta * np.sqrt(8 / np.pi)
    R2 = D ** 2 / (D ** 2 + np.pi ** 2 / 3)
    return float(D), float(R2)


def bootstrap_ci(metric_fn, arrays, n_boot=200, seed=42):
    rng = np.random.RandomState(seed)
    n = len(arrays[0])
    point = metric_fn(*arrays)
    vals = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        try:
            v = metric_fn(*[a[idx] for a in arrays])
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    if len(vals) < 10:
        return point, np.nan, np.nan
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


# ====================================================================
# Data Loading & Episode Construction
# ====================================================================
def load_data():
    print("[INFO] Loading data...")
    df = pd.read_csv(IMPUTED_PATH, low_memory=False)
    split_map = pd.read_csv(SPLIT_PATH)
    df = df.merge(split_map, on="id", how="left")

    df["time_years"] = pd.to_numeric(df["time"], errors="coerce")
    bad = df["time_years"].isna() | (df["time_years"] <= 0)
    if "t_start" in df.columns and "t_stop" in df.columns:
        t0 = pd.to_datetime(df["t_start"], errors="coerce")
        t1 = pd.to_datetime(df["t_stop"], errors="coerce")
        delta = (t1 - t0).dt.days / 365.25
        df.loc[bad, "time_years"] = delta[bad]

    df["state_start"] = pd.to_numeric(df["state_start"], errors="coerce")
    df["state_stop"] = pd.to_numeric(df["state_stop"], errors="coerce")
    df = df[df["state_start"].isin([0, 1, 2]) & df["state_stop"].isin([0, 1, 2])]
    df = df[df["time_years"].notna() & (df["time_years"] > 0)]
    df = df[df["split"].isin(["train", "val", "test"])]
    print(f"  {len(df)} rows, {df['id'].nunique()} IDs")
    return df


def build_episodes(df, from_state=0):
    """Collapse counting-process data to one episode per person (from given state)."""
    print(f"[INFO] Building episodes from state {from_state}...")
    all_feats = list(CONTINUOUS_VARS | CATEGORICAL_VARS)
    all_feats = [c for c in all_feats if c in df.columns]
    recs = []
    for pid, grp in df.groupby("id"):
        grp = grp.sort_values("time_years").reset_index(drop=True)
        split_val = grp["split"].iloc[0]
        n = len(grp)
        i = 0
        while i < n:
            if int(grp.iloc[i]["state_start"]) != from_state:
                i += 1
                continue
            start_i = i
            duration = 0.0
            status = 0
            j = i
            while True:
                duration += grp.iloc[j]["time_years"]
                to_state = int(grp.iloc[j]["state_stop"])
                if to_state != from_state:
                    status = 1 if to_state == 1 else 2
                    break
                j += 1
                if j >= n or int(grp.iloc[j]["state_start"]) != from_state:
                    break
            base = grp.iloc[start_i]
            rec = {"id": pid, "split": split_val, "duration": duration, "status": int(status)}
            for c in all_feats:
                rec[c] = base.get(c, np.nan)
            recs.append(rec)
            break  # first episode per person only
    episodes = pd.DataFrame(recs)
    episodes = episodes[episodes["duration"] > 0.01].reset_index(drop=True)
    sc = episodes["status"].value_counts().sort_index()
    print(f"  {len(episodes)} episodes — status: {dict(sc)}")
    return episodes


def build_episodes_combined(df):
    """One episode per person from first segment with state_start in (0, 2). Status: 1=T2D, 2=PreDM."""
    print("[INFO] Building episodes (state 0 + state 2 combined)...")
    all_feats = list(CONTINUOUS_VARS | CATEGORICAL_VARS)
    all_feats = [c for c in all_feats if c in df.columns]
    recs = []
    for pid, grp in df.groupby("id"):
        grp = grp.sort_values("time_years").reset_index(drop=True)
        split_val = grp["split"].iloc[0]
        n = len(grp)
        i = 0
        while i < n:
            s_start = int(grp.iloc[i]["state_start"])
            if s_start not in (0, 2):
                i += 1
                continue
            start_i = i
            duration = 0.0
            status = 0
            j = i
            while True:
                duration += grp.iloc[j]["time_years"]
                to_state = int(grp.iloc[j]["state_stop"])
                if to_state != s_start:
                    status = 1 if to_state == 1 else (2 if (s_start == 0 and to_state == 2) else 0)
                    break
                j += 1
                if j >= n or int(grp.iloc[j]["state_start"]) != s_start:
                    break
            base = grp.iloc[start_i]
            rec = {"id": pid, "split": split_val, "duration": duration, "status": int(status), "state_start": s_start}
            for c in all_feats:
                rec[c] = base.get(c, np.nan)
            recs.append(rec)
            break
    episodes = pd.DataFrame(recs)
    episodes = episodes[episodes["duration"] > 0.01].reset_index(drop=True)
    sc = episodes["status"].value_counts().sort_index()
    print(f"  {len(episodes)} episodes — status: {dict(sc)}")
    return episodes


# ====================================================================
# Summary tables
# ====================================================================
def _rate_ci(events, py):
    """Incidence per 1000 person-years with 95 % Poisson CI."""
    if py < 0.001:
        return "—"
    rate = events / py * 1000
    if events == 0:
        hi = chi2.ppf(0.975, 2) / 2 / py * 1000
        return f"0.00 (0.00 to {hi:.2f})"
    lo = chi2.ppf(0.025, 2 * events) / 2 / py * 1000
    hi = chi2.ppf(0.975, 2 * (events + 1)) / 2 / py * 1000
    return f"{rate:.2f} ({lo:.2f} to {hi:.2f})"


def _fmt_mean_sd(series):
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "NA"
    return f"{x.mean():.3f} ± {x.std(ddof=1):.3f}"


def _fmt_count_pct(series, level):
    x = pd.to_numeric(series, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return "NA"
    n = int((x == level).sum())
    pct = 100.0 * n / len(x)
    return f"{n} ({pct:.1f}%)"


def build_feature_distribution_table(train_ep, val_ep, test_ep):
    split_map = {"Train": train_ep, "Validation": val_ep, "Test": test_ep}
    rows = []
    continuous_order = ["age", "lifestyle_score", "weight", "DBP", "HR", "SBP", "FPG", "HbA1c"]
    categorical_order = [
        ("CVD", [0, 1]),
        ("ethnichan", [0, 1]),
        ("fam_diabetes", [0, 1]),
        ("hyperlipid", [0, 1]),
        ("hypertension", [0, 1]),
        ("shift_work", [0, 1]),
    ]

    for var in continuous_order:
        if var not in train_ep.columns:
            continue
        row = {"Variable": var, "Level": "continuous"}
        for split_name, df in split_map.items():
            row[split_name] = _fmt_mean_sd(df[var])
        rows.append(row)

    for var, levels in categorical_order:
        if var not in train_ep.columns:
            continue
        for idx, level in enumerate(levels):
            row = {"Variable": var if idx == 0 else "", "Level": str(level)}
            for split_name, df in split_map.items():
                row[split_name] = _fmt_count_pct(df[var], level)
            rows.append(row)

    return pd.DataFrame(rows)


def build_grouped_incidence_table(episodes):
    train_ep = episodes[episodes["split"] == "train"].copy()
    medians = {
        var: pd.to_numeric(train_ep[var], errors="coerce").median()
        for var in ["DBP", "FPG", "HR", "HbA1c", "SBP", "lifestyle_score", "weight"]
        if var in train_ep.columns
    }

    age_groups = [
        ("age <50", lambda d: pd.to_numeric(d["age"], errors="coerce") < 50),
        ("age 50-54", lambda d: (pd.to_numeric(d["age"], errors="coerce") >= 50) & (pd.to_numeric(d["age"], errors="coerce") < 55)),
        ("age 55-59", lambda d: (pd.to_numeric(d["age"], errors="coerce") >= 55) & (pd.to_numeric(d["age"], errors="coerce") < 60)),
        ("age 60-64", lambda d: (pd.to_numeric(d["age"], errors="coerce") >= 60) & (pd.to_numeric(d["age"], errors="coerce") < 65)),
        ("age 65-69", lambda d: (pd.to_numeric(d["age"], errors="coerce") >= 65) & (pd.to_numeric(d["age"], errors="coerce") < 70)),
        ("age 70-74", lambda d: (pd.to_numeric(d["age"], errors="coerce") >= 70) & (pd.to_numeric(d["age"], errors="coerce") < 75)),
        ("age >=75", lambda d: pd.to_numeric(d["age"], errors="coerce") >= 75),
    ]

    group_defs = [
        ("Total", lambda d: pd.Series(True, index=d.index)),
        ("CVD=0", lambda d: pd.to_numeric(d["CVD"], errors="coerce") == 0),
        ("CVD=1", lambda d: pd.to_numeric(d["CVD"], errors="coerce") == 1),
        ("DBP ≤median", lambda d: pd.to_numeric(d["DBP"], errors="coerce") <= medians["DBP"]),
        ("DBP >median", lambda d: pd.to_numeric(d["DBP"], errors="coerce") > medians["DBP"]),
        ("FPG ≤median", lambda d: pd.to_numeric(d["FPG"], errors="coerce") <= medians["FPG"]),
        ("FPG >median", lambda d: pd.to_numeric(d["FPG"], errors="coerce") > medians["FPG"]),
        ("HR ≤median", lambda d: pd.to_numeric(d["HR"], errors="coerce") <= medians["HR"]),
        ("HR >median", lambda d: pd.to_numeric(d["HR"], errors="coerce") > medians["HR"]),
        ("HbA1c ≤median", lambda d: pd.to_numeric(d["HbA1c"], errors="coerce") <= medians["HbA1c"]),
        ("HbA1c >median", lambda d: pd.to_numeric(d["HbA1c"], errors="coerce") > medians["HbA1c"]),
        ("SBP ≤median", lambda d: pd.to_numeric(d["SBP"], errors="coerce") <= medians["SBP"]),
        ("SBP >median", lambda d: pd.to_numeric(d["SBP"], errors="coerce") > medians["SBP"]),
    ] + age_groups + [
        ("ethnichan=0", lambda d: pd.to_numeric(d["ethnichan"], errors="coerce") == 0),
        ("ethnichan=1", lambda d: pd.to_numeric(d["ethnichan"], errors="coerce") == 1),
        ("fam_diabetes=0", lambda d: pd.to_numeric(d["fam_diabetes"], errors="coerce") == 0),
        ("fam_diabetes=1", lambda d: pd.to_numeric(d["fam_diabetes"], errors="coerce") == 1),
        ("hyperlipid=0", lambda d: pd.to_numeric(d["hyperlipid"], errors="coerce") == 0),
        ("hyperlipid=1", lambda d: pd.to_numeric(d["hyperlipid"], errors="coerce") == 1),
        ("hypertension=0", lambda d: pd.to_numeric(d["hypertension"], errors="coerce") == 0),
        ("hypertension=1", lambda d: pd.to_numeric(d["hypertension"], errors="coerce") == 1),
        ("lifestyle_score ≤median", lambda d: pd.to_numeric(d["lifestyle_score"], errors="coerce") <= medians["lifestyle_score"]),
        ("lifestyle_score >median", lambda d: pd.to_numeric(d["lifestyle_score"], errors="coerce") > medians["lifestyle_score"]),
        ("shift_work=0", lambda d: pd.to_numeric(d["shift_work"], errors="coerce") == 0),
        ("shift_work=1", lambda d: pd.to_numeric(d["shift_work"], errors="coerce") == 1),
        ("weight ≤median", lambda d: pd.to_numeric(d["weight"], errors="coerce") <= medians["weight"]),
        ("weight >median", lambda d: pd.to_numeric(d["weight"], errors="coerce") > medians["weight"]),
    ]

    split_map = {
        "Train": episodes[episodes["split"] == "train"],
        "Validation": episodes[episodes["split"] == "val"],
        "Test": episodes[episodes["split"] == "test"],
    }
    outcome_map = {1: "T2D", 2: "PreDM"}
    rows = []
    for group_name, mask_fn in group_defs:
        row = {"Group": group_name}
        for split_name, df in split_map.items():
            grp = df.loc[mask_fn(df)].copy()
            py = grp["duration"].sum()
            row[f"{split_name}_N"] = len(grp)
            row[f"{split_name}_Person_years_k"] = round(py / 1000, 3)
            for outcome_code, outcome_name in outcome_map.items():
                events = int((grp["status"] == outcome_code).sum())
                row[f"{split_name}_{outcome_name}_cases"] = events
                row[f"{split_name}_{outcome_name}_Incidence_per_1000_95CI"] = _rate_ci(events, py)
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================
# Model fitting pipeline (one model config)
# ====================================================================
def _impute_features(train, val, test, features):
    """Fill NaN with training median/mode."""
    for f in features:
        if f not in train.columns:
            continue
        col = train[f]
        if f in CONTINUOUS_VARS:
            fill = col.median()
        else:
            fill = col.mode().iloc[0] if len(col.mode()) else 0
        for d in (train, val, test):
            d[f] = d[f].fillna(fill)


def fit_model(model_name, features, train_ep, val_ep, test_ep):
    """Full pipeline: FP → VIF → Cox → PH test → Age interactions → Recal → CIF → Metrics."""
    print(f"\n{'='*70}\n  {model_name}\n{'='*70}")
    cont = [f for f in features if f in CONTINUOUS_VARS]
    cat = [f for f in features if f in CATEGORICAL_VARS]
    ph_records = []
    age_records = []

    tr = train_ep.copy()
    va = val_ep.copy()
    te = test_ep.copy()
    _impute_features(tr, va, te, features)

    # --- 1. FP selection (on training, using cause-1 event) ---
    print("[1] Fractional Polynomial selection...")
    tr["_evt1"] = (tr["status"] == 1).astype(int)
    fp_specs = {}
    for var in cont:
        print(f"    {var} ... ", end="", flush=True)
        spec = select_fp_for_variable(tr, var, features, "duration", "_evt1")
        fp_specs[var] = spec
        print(f"{spec['type']} p={spec['powers']}")
    tr.drop(columns=["_evt1"], inplace=True, errors="ignore")

    tr, feat_list, fp_stats, fp_groups = apply_fp_transforms(tr, cont, cat, fp_specs)
    va, _, _, _ = apply_fp_transforms(va, cont, cat, fp_specs, fit_stats=fp_stats)
    te, _, _, _ = apply_fp_transforms(te, cont, cat, fp_specs, fit_stats=fp_stats)

    # --- 2. VIF ---
    print("[2] VIF check...")
    feat_list, removed = check_vif(tr, feat_list, fp_groups, threshold=5.0)
    print(f"    Kept {len(feat_list)} features")

    # --- 3-6: Cause-specific Cox models ---
    cause_info = {}
    for cause in [1, 2]:
        tag = "T2D" if cause == 1 else "PreDM"
        print(f"\n[3-6] Cause {cause} ({tag}) ...")
        evt_col = f"_evt_c{cause}"
        for d in (tr, va, te):
            d[evt_col] = (d["status"] == cause).astype(int)

        model_feats = list(feat_list)
        fit_df = tr[model_feats + ["duration", evt_col]].dropna()
        try:
            cph = CoxPHFitter(penalizer=0.01)
            cph.fit(fit_df, duration_col="duration", event_col=evt_col, show_progress=False)
        except Exception as e:
            print(f"  Initial fit FAILED: {e}")
            continue

        # Schoenfeld test
        time_int_vars = []
        try:
            ph = proportional_hazard_test(cph, fit_df, time_transform="rank")
            for vn in ph.summary.index:
                vn_str = vn if isinstance(vn, str) else vn[0]
                pv = ph.summary.loc[vn, "p"]
                int_col = ""
                if pv < 0.05:
                    int_col = f"{vn_str}_x_logt"
                    for d in (tr, va, te):
                        d[int_col] = d[vn_str] * np.log(np.maximum(d["duration"], 0.01))
                    time_int_vars.append(int_col)
                    print(f"    PH violated: {vn_str} (p={pv:.4f}) → added {int_col}")
                ph_records.append({
                    "Model": model_name,
                    "Cause": tag,
                    "Variable": vn_str,
                    "p_value": float(pv),
                    "PH_violated": "Yes" if pv < 0.05 else "No",
                    "Interaction_added": int_col,
                })
        except Exception:
            pass

        # Age interactions: test on top of already identified time interactions,
        # and keep previously retained age interactions in the screening model.
        age_col = "age" if "age" in model_feats else next((f for f in model_feats if f.startswith("age")), None)
        age_int_vars = []
        if age_col:
            for var in model_feats:
                if var == age_col:
                    continue
                ic = f"age_x_{var}"
                for d in (tr, va, te):
                    d[ic] = d[age_col] * d[var]
                test_feats = model_feats + time_int_vars + age_int_vars + [ic]
                tdf = tr[test_feats + ["duration", evt_col]].dropna()
                try:
                    ct = CoxPHFitter(penalizer=0.01)
                    ct.fit(tdf, duration_col="duration", event_col=evt_col, show_progress=False)
                    pv = ct.summary.loc[ic, "p"]
                    age_records.append({
                        "Model": model_name,
                        "Cause": tag,
                        "Age_term": age_col,
                        "Variable": var,
                        "Interaction_term": ic,
                        "p_value": float(pv),
                        "Retained": "Yes" if pv < 0.05 else "No",
                    })
                    if pv < 0.05:
                        age_int_vars.append(ic)
                        print(f"    Age interaction: {var} (p={pv:.4f})")
                except Exception:
                    pass

        final_feats = model_feats + time_int_vars + age_int_vars
        final_groups = dict(fp_groups)
        final_groups.update({feat: [feat] for feat in time_int_vars + age_int_vars})
        post_removed = []
        if len(final_feats) > len(model_feats):
            print("[6b] Post-interaction VIF check...")
            final_feats, post_removed = check_vif(
                tr,
                final_feats,
                final_groups,
                threshold=5.0,
                prefer_interactions=True,
            )
            removed.extend(post_removed)
        time_int_vars = [f for f in time_int_vars if f in final_feats]
        age_int_vars = [f for f in age_int_vars if f in final_feats]
        for rec in ph_records:
            if rec["Model"] == model_name and rec["Cause"] == tag and rec["Interaction_added"]:
                if rec["Interaction_added"] not in time_int_vars:
                    rec["Interaction_added"] = f"{rec['Interaction_added']} [removed_by_vif]"
        for rec in age_records:
            if rec["Model"] == model_name and rec["Cause"] == tag and rec["Interaction_term"]:
                rec["Retained"] = "Yes" if rec["Interaction_term"] in age_int_vars else "No"
        fdf = tr[final_feats + ["duration", evt_col]].dropna()
        try:
            cph_final = CoxPHFitter(penalizer=0.01)
            cph_final.fit(fdf, duration_col="duration", event_col=evt_col, show_progress=False)
        except Exception:
            cph_final = cph
            final_feats = model_feats
            time_int_vars = []
            age_int_vars = []

        print(f"    Train C-index: {cph_final.concordance_index_:.4f}")

        # Recalibrate on validation
        val_X = va[final_feats].fillna(0)
        lp_val = cph_final.predict_log_partial_hazard(val_X).values.flatten()
        cal_slope = 1.0
        try:
            recal_df = pd.DataFrame({"dur": va["duration"], "evt": va[evt_col], "lp": lp_val}).dropna()
            recal_df = recal_df[recal_df["dur"] > 0]
            rc = CoxPHFitter(penalizer=0.0)
            rc.fit(recal_df, duration_col="dur", event_col="evt", show_progress=False)
            cal_slope = float(rc.params_["lp"])
            print(f"    Calibration slope: {cal_slope:.4f}")
        except Exception:
            pass

        adj_lp_val = cal_slope * lp_val[: len(recal_df)] if "recal_df" in dir() else cal_slope * lp_val
        baseline_df = breslow_baseline(
            recal_df["dur"].values if "recal_df" in dir() else va["duration"].values,
            (recal_df["evt"].values if "recal_df" in dir() else va[evt_col].values).astype(int),
            adj_lp_val,
        )

        # Separate time-fixed and time-varying LP parts for test set
        base_feats = [f for f in final_feats if not f.endswith("_x_logt")]
        ti_feats = [f for f in final_feats if f.endswith("_x_logt")]
        ti_base_vars = [f.replace("_x_logt", "") for f in ti_feats]

        base_lp = np.zeros(len(te))
        for feat in base_feats:
            if feat in cph_final.params_.index:
                base_lp += te[feat].fillna(0).values * cph_final.params_[feat]
        base_lp *= cal_slope

        # lifelines predicts centered log-partial-hazards: (x - mean_train) * beta.
        # Keep test-time CIF prediction on the same scale as the validation-time
        # baseline hazard estimation by subtracting the training centering offset.
        lp_center = 0.0
        norm_mean = getattr(cph_final, "_norm_mean", pd.Series(dtype=float))
        for feat in final_feats:
            if feat in cph_final.params_.index and feat in norm_mean.index:
                lp_center += float(norm_mean[feat] * cph_final.params_[feat])
        base_lp -= lp_center * cal_slope

        ti_data, ti_beta = None, None
        if ti_feats:
            ti_data = te[ti_base_vars].fillna(0).values
            ti_beta = np.array([cph_final.params_[f] * cal_slope for f in ti_feats])

        cause_info[cause] = {
            "fitter": cph_final,
            "features": final_feats,
            "cal_slope": cal_slope,
            "baseline": baseline_df,
            "base_lp": base_lp,
            "ti_data": ti_data,
            "ti_beta": ti_beta,
            "time_int_vars": time_int_vars,
            "age_int_vars": age_int_vars,
        }

    if 1 not in cause_info or 2 not in cause_info:
        print("  WARN: Could not fit both cause models.")
        return None

    # --- CIF on test set ---
    print("\n[7] Computing CIF on test set...")
    c1 = cause_info[1]
    c2 = cause_info[2]
    cif_res = compute_cif(
        c1["base_lp"], c2["base_lp"],
        c1["baseline"], c2["baseline"],
        c1["ti_data"], c1["ti_beta"],
        c2["ti_data"], c2["ti_beta"],
        max_time=max(HORIZONS.values()),
    )

    risk_c1 = interpolate_cif_at_horizon(cif_res, 1, HORIZONS[1])
    risk_c2 = interpolate_cif_at_horizon(cif_res, 2, HORIZONS[2])

    # --- Evaluation ---
    print("[8] Evaluation metrics...")
    G_func = make_G_func(tr["duration"].values, tr["status"].values)
    test_t = te["duration"].values
    test_s = te["status"].values.astype(int)

    metrics = {}
    for cause, horizon, risk, tag in [
        (1, HORIZONS[1], risk_c1, "T2D_10y"),
        (2, HORIZONS[2], risk_c2, "PreDM_5y"),
    ]:
        evt_bin = (test_s == cause).astype(int)

        c_idx, c_lo, c_hi = bootstrap_ci(
            lambda t, r, e: concordance_index(t, -r, e),
            [test_t, risk, evt_bin],
        )
        auc_val, a_lo, a_hi = bootstrap_ci(
            lambda t, s, r: td_auc(t, s, r, cause, horizon),
            [test_t, test_s, risk],
        )
        bs_val, bs_lo, bs_hi = bootstrap_ci(
            lambda t, s, r: ipcw_brier(t, s, r, cause, horizon, G_func),
            [test_t, test_s, risk],
        )
        cif_mat = cif_res["cif1"] if cause == 1 else cif_res["cif2"]
        ibs_val, ibs_lo, ibs_hi = bootstrap_ci(
            lambda t, s, cm: integrated_brier_score(t, s, cif_res["times"], cm, cause, horizon, G_func),
            [test_t, test_s, cif_mat],
        )

        lp_cause = c1["base_lp"] if cause == 1 else c2["base_lp"]
        D_val, R2_val = royston_d(test_t, evt_bin, lp_cause)
        D_pt, D_lo, D_hi = bootstrap_ci(
            lambda t, e, l: royston_d(t, e, l)[0],
            [test_t, evt_bin, lp_cause],
        )
        _, R2_lo, R2_hi = bootstrap_ci(
            lambda t, e, l: royston_d(t, e, l)[1],
            [test_t, evt_bin, lp_cause],
        )

        calib_slope = horizon_logistic_calibration_slope(test_t, test_s, risk, cause, horizon)

        metrics[tag] = {
            "C_index": f"{c_idx:.3f} ({c_lo:.3f} to {c_hi:.3f})",
            "AUC": f"{auc_val:.3f} ({a_lo:.3f} to {a_hi:.3f})",
            "Brier": f"{bs_val:.4f} ({bs_lo:.4f} to {bs_hi:.4f})" if np.isfinite(bs_val) else "NA",
            "IBS": f"{ibs_val:.4f} ({ibs_lo:.4f} to {ibs_hi:.4f})" if np.isfinite(ibs_val) else "NA",
            "D_statistic": f"{D_val:.2f} ({D_lo:.2f} to {D_hi:.2f})" if np.isfinite(D_val) else "NA",
            "R2_pct": f"{R2_val*100:.1f} ({R2_lo*100:.1f} to {R2_hi*100:.1f})" if np.isfinite(R2_val) else "NA",
            "Calibration_slope": f"{calib_slope:.3f}" if np.isfinite(calib_slope) else "NA",
        }
        print(f"  {tag}: C={c_idx:.3f}, AUC={auc_val:.3f}, Brier={bs_val:.4f}")

    return {
        "model_name": model_name,
        "cause_info": cause_info,
        "cif_res": cif_res,
        "risk_c1": risk_c1,
        "risk_c2": risk_c2,
        "metrics": metrics,
        "test_t": test_t,
        "test_s": test_s,
        "fp_specs": fp_specs,
        "vif_removed": removed,
        "ph_records": ph_records,
        "age_records": age_records,
    }


# ====================================================================
# HR table
# ====================================================================
def _hr_union_vars():
    model_order = ["Model1", "Model2", "Model3", "Model4"]
    union_vars = []
    for m in model_order:
        for v in MODEL_FEATURES.get(m, []):
            if v not in union_vars:
                union_vars.append(v)
    return union_vars


def _hr_levels_for_var(var, train_series=None):
    if var == "lifestyle_score":
        return ["6", "5", "4", "3", "2", "1", "0"], "6"
    if var in CONTINUOUS_VARS:
        return ["Q1", "Q2", "Q3", "Q4"], "Q1"
    s = pd.to_numeric(train_series, errors="coerce") if train_series is not None else pd.Series(dtype=float)
    uniq = sorted([x for x in pd.unique(s.dropna()) if np.isfinite(x)])
    if len(uniq) == 0:
        uniq = [0, 1]
    ref = 0 if 0 in uniq else uniq[0]
    levels = [str(int(x)) if float(x).is_integer() else str(x) for x in uniq]
    ref_label = str(int(ref)) if float(ref).is_integer() else str(ref)
    return levels, ref_label


def _assign_quartile_labels(series):
    x = pd.to_numeric(series, errors="coerce")
    if x.notna().sum() < 4:
        return pd.Series(["Q1"] * len(x), index=series.index, dtype=object)
    ranked = x.rank(method="first")
    q = pd.qcut(ranked, 4, labels=["Q1", "Q2", "Q3", "Q4"])
    return q.astype(str)


def _prepare_hr_design(train_ep, features, cause):
    df = train_ep[["duration", "status"] + features].copy()
    evt_col = "_event"
    df[evt_col] = (df["status"] == cause).astype(int)

    for f in features:
        col = pd.to_numeric(df[f], errors="coerce") if f in CONTINUOUS_VARS or f in CATEGORICAL_VARS else df[f]
        if f in CONTINUOUS_VARS:
            fill = pd.to_numeric(df[f], errors="coerce").median()
            df[f] = pd.to_numeric(df[f], errors="coerce").fillna(fill)
        else:
            s = pd.to_numeric(df[f], errors="coerce")
            mode = s.mode().iloc[0] if len(s.mode()) else 0
            df[f] = s.fillna(mode)

    design = pd.DataFrame(index=df.index)
    variable_cells = {}
    variable_rows = {}

    for var in features:
        if var == "lifestyle_score":
            vals = pd.to_numeric(df[var], errors="coerce").round().astype(int)
            levels, ref = _hr_levels_for_var(var, vals)
            variable_rows[var] = levels
            variable_cells[(var, ref)] = "Ref"
            for lv in levels:
                if lv == ref:
                    continue
                col_name = f"{var}__{lv}"
                design[col_name] = (vals == int(lv)).astype(int)
                variable_cells[(var, lv)] = col_name
        elif var in CONTINUOUS_VARS:
            q = _assign_quartile_labels(df[var])
            levels, ref = _hr_levels_for_var(var, df[var])
            variable_rows[var] = levels
            variable_cells[(var, ref)] = "Ref"
            for lv in levels:
                if lv == ref:
                    continue
                col_name = f"{var}__{lv}"
                design[col_name] = (q == lv).astype(int)
                variable_cells[(var, lv)] = col_name
        else:
            vals = pd.to_numeric(df[var], errors="coerce")
            levels, ref = _hr_levels_for_var(var, vals)
            variable_rows[var] = levels
            variable_cells[(var, ref)] = "Ref"
            for lv in levels:
                if lv == ref:
                    continue
                target = float(lv)
                col_name = f"{var}__{lv}"
                design[col_name] = (vals == target).astype(int)
                variable_cells[(var, lv)] = col_name

    fit_df = pd.concat([df[["duration", evt_col]], design], axis=1)
    return fit_df, evt_col, variable_cells, variable_rows


def _fit_adjusted_hr_categorical(train_ep, features, cause):
    fit_df, evt_col, variable_cells, variable_rows = _prepare_hr_design(train_ep, features, cause)
    predictors = [c for c in fit_df.columns if c not in ["duration", evt_col]]
    if len(predictors) == 0:
        return {}, variable_rows
    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(fit_df[["duration", evt_col] + predictors], duration_col="duration", event_col=evt_col, show_progress=False)

    out = {}
    for (var, level), ref_or_col in variable_cells.items():
        if ref_or_col == "Ref":
            out[(var, level)] = "Ref"
            continue
        if ref_or_col not in cph.summary.index:
            out[(var, level)] = "-"
            continue
        s = cph.summary.loc[ref_or_col]
        hr = float(np.exp(s["coef"]))
        lo = float(np.exp(s["coef"] - 1.96 * s["se(coef)"]))
        hi = float(np.exp(s["coef"] + 1.96 * s["se(coef)"]))
        out[(var, level)] = f"{hr:.3f}({lo:.3f} to {hi:.3f})"
    return out, variable_rows


def build_hr_table(train_ep):
    model_order = ["Model1", "Model2", "Model3", "Model4"]
    union_vars = _hr_union_vars()
    hr_maps = {}
    row_defs = {}

    for cause in [1, 2]:
        for model_name in model_order:
            features = MODEL_FEATURES.get(model_name, [])
            hr_maps[(cause, model_name)], var_rows = _fit_adjusted_hr_categorical(train_ep, features, cause)
            for v, levels in var_rows.items():
                row_defs[v] = levels

    rows = []
    for cause in [1, 2]:
        cause_tag = "T2D" if cause == 1 else "PreDM"
        for var in union_vars:
            levels = row_defs.get(var)
            if levels is None:
                levels, _ = _hr_levels_for_var(var, train_ep[var] if var in train_ep.columns else None)
            for i, level in enumerate(levels):
                row = {
                    "Cause": cause_tag,
                    "Variable": var if i == 0 else "",
                    "Level": level,
                }
                for model_name in model_order:
                    if var not in MODEL_FEATURES.get(model_name, []):
                        row[model_name] = "-"
                    else:
                        row[model_name] = hr_maps.get((cause, model_name), {}).get((var, level), "-")
                rows.append(row)

    return pd.DataFrame(rows, columns=["Cause", "Variable", "Level"] + model_order)


# ====================================================================
# Performance table (like 图片5)
# ====================================================================
def build_performance_table(all_results):
    rows = []
    stat_names = ["D_statistic", "C_index", "R2_pct", "AUC", "Brier", "IBS", "Calibration_slope"]
    for stat in stat_names:
        row = {"Statistic": stat}
        for res in all_results:
            if res is None:
                continue
            for tag in ["T2D_10y", "PreDM_5y"]:
                col = f"{res['model_name']}_{tag}"
                row[col] = res["metrics"].get(tag, {}).get(stat, "NA")
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================
# Modeling report tables
# ====================================================================
def build_modeling_report_tables(all_results):
    fp_rows = []
    vif_rows = []
    ph_rows = []
    age_rows = []

    for res in all_results:
        if res is None:
            continue
        model_name = res["model_name"]
        for var, spec in res["fp_specs"].items():
            fp_rows.append({
                "Model": model_name,
                "Variable": var,
                "Selected_form": spec["type"],
                "Powers": ",".join(str(x) for x in spec["powers"]),
                "Shift": spec["shift"],
            })
        for item in res.get("vif_removed", []):
            vif_rows.append({
                "Model": model_name,
                "Removed_feature": item["feature"],
                "Original_variable": item["original_variable"],
                "VIF": item["vif"],
            })
        ph_rows.extend(res.get("ph_records", []))
        age_rows.extend(res.get("age_records", []))

    fp_df = pd.DataFrame(fp_rows, columns=["Model", "Variable", "Selected_form", "Powers", "Shift"])
    vif_df = pd.DataFrame(vif_rows, columns=["Model", "Removed_feature", "Original_variable", "VIF"])
    ph_df = pd.DataFrame(
        ph_rows,
        columns=["Model", "Cause", "Variable", "p_value", "PH_violated", "Interaction_added"],
    )
    age_df = pd.DataFrame(
        age_rows,
        columns=["Model", "Cause", "Age_term", "Variable", "Interaction_term", "p_value", "Retained"],
    )
    return fp_df, vif_df, ph_df, age_df


# ====================================================================
# Calibration plot
# ====================================================================
def plot_calibration(all_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (cause, horizon, title) in enumerate([
        (1, HORIZONS[1], "10年T2D累积发病风险校准曲线"),
        (2, HORIZONS[2], "5年糖调节受损累积风险校准曲线"),
    ]):
        ax = axes[ax_idx]
        for res in all_results:
            if res is None:
                continue
            risk = res["risk_c1"] if cause == 1 else res["risk_c2"]
            t, s = res["test_t"], res["test_s"]
            try:
                cuts = pd.qcut(risk, 10, labels=False, duplicates="drop")
            except Exception:
                cuts = pd.cut(risk, 10, labels=False)
            pred_means, obs_means = [], []
            for g in sorted(np.unique(cuts[~np.isnan(cuts)])):
                mask = cuts == g
                pred_means.append(risk[mask].mean() * 100)
                obs_means.append(aj_cif(t[mask], s[mask], cause, horizon) * 100)
            ax.plot(pred_means, obs_means, "o-", label=res["model_name"], markersize=4)
        lim = ax.get_xlim()[1]
        ax.plot([0, lim], [0, lim], "--", color="grey", linewidth=0.8)
        ax.set_xlabel("预测风险百分比")
        ax.set_ylabel("观测风险百分比")
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "calibration_plot.png"), dpi=200)
    plt.close()
    print(f"  Saved calibration_plot.png")


# ====================================================================
# Decision Curve Analysis
# ====================================================================
def plot_dca(all_results, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax_idx, (cause, horizon, title) in enumerate([
        (1, HORIZONS[1], "10年T2D累积发病风险决策曲线"),
        (2, HORIZONS[2], "5年糖调节受损累积风险决策曲线"),
    ]):
        ax = axes[ax_idx]
        thresholds = np.arange(0.01, 0.51, 0.01)

        for res in all_results:
            if res is None:
                continue
            risk = res["risk_c1"] if cause == 1 else res["risk_c2"]
            t, s = res["test_t"], res["test_s"]
            y = ((t <= horizon) & (s == cause)).astype(int)
            keep = ((t <= horizon) & (s != 0)) | (t > horizon)
            y_k, r_k = y[keep], risk[keep]
            n = len(y_k)
            prev = y_k.mean()
            nb = []
            for pt in thresholds:
                tp = np.sum((r_k >= pt) & (y_k == 1))
                fp = np.sum((r_k >= pt) & (y_k == 0))
                nb.append(tp / n - fp / n * (pt / (1 - pt)))
            ax.plot(thresholds, nb, label=res["model_name"], linewidth=1)

        # Treat All / Treat None
        if all_results and all_results[0] is not None:
            res0 = all_results[0]
            risk0 = res0["risk_c1"] if cause == 1 else res0["risk_c2"]
            t0, s0 = res0["test_t"], res0["test_s"]
            y0 = ((t0 <= horizon) & (s0 == cause)).astype(int)
            keep0 = ((t0 <= horizon) & (s0 != 0)) | (t0 > horizon)
            prev0 = y0[keep0].mean()
            nb_all = [prev0 - (1 - prev0) * (pt / (1 - pt)) for pt in thresholds]
            ax.plot(thresholds, nb_all, "--", color="grey", label="Treat all")
            ax.axhline(0, color="black", linewidth=0.5, label="Treat none")

        ax.set_xlabel("阈值概率")
        ax.set_ylabel("净收益")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "dca_plot.png"), dpi=200)
    plt.close()
    print(f"  Saved dca_plot.png")


# ====================================================================
# Main
# ====================================================================
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load & build episodes
    raw = load_data()
    episodes = build_episodes(raw, from_state=0)

    train_ep = episodes[episodes["split"] == "train"].reset_index(drop=True)
    val_ep = episodes[episodes["split"] == "val"].reset_index(drop=True)
    test_ep = episodes[episodes["split"] == "test"].reset_index(drop=True)
    print(f"  Train: {len(train_ep)}, Val: {len(val_ep)}, Test: {len(test_ep)}")

    # Grouped incidence table
    print("\n[INFO] Grouped incidence table...")
    incidence_df = build_grouped_incidence_table(episodes)
    incidence_df.to_csv(os.path.join(OUT_DIR, "table_incidence_by_group.csv"), index=False)
    print("  Saved table_incidence_by_group.csv")

    # Feature distribution table
    print("\n[INFO] Feature distribution table...")
    feat_dist = build_feature_distribution_table(train_ep, val_ep, test_ep)
    feat_dist.to_csv(os.path.join(OUT_DIR, "table_feature_distribution.csv"), index=False)
    print("  Saved table_feature_distribution.csv")

    # --- State 0 + State 2 combined series ---
    episodes_combined = build_episodes_combined(raw)
    train_comb = episodes_combined[episodes_combined["split"] == "train"].reset_index(drop=True)
    val_comb = episodes_combined[episodes_combined["split"] == "val"].reset_index(drop=True)
    test_comb = episodes_combined[episodes_combined["split"] == "test"].reset_index(drop=True)
    print(f"  Combined Train: {len(train_comb)}, Val: {len(val_comb)}, Test: {len(test_comb)}")
    print("\n[INFO] Combined (state 0+2) feature distribution table...")
    feat_dist_comb = build_feature_distribution_table(train_comb, val_comb, test_comb)
    feat_dist_comb.to_csv(os.path.join(OUT_DIR, "table_feature_distribution_combined.csv"), index=False)
    print("  Saved table_feature_distribution_combined.csv")
    print("\n[INFO] Combined (state 0+2) event incidence table...")
    incidence_comb = build_grouped_incidence_table(episodes_combined)
    incidence_comb.to_csv(os.path.join(OUT_DIR, "table_incidence_by_group_combined.csv"), index=False)
    print("  Saved table_incidence_by_group_combined.csv")

    # Fit models
    all_results = []
    for model_name, features in MODEL_FEATURES.items():
        res = fit_model(model_name, features, train_ep, val_ep, test_ep)
        all_results.append(res)

    # HR table
    print("\n[INFO] HR table...")
    hr_df = build_hr_table(train_ep)
    hr_df.to_csv(os.path.join(OUT_DIR, "table_hr.csv"), index=False)
    print(f"  Saved table_hr.csv")

    # Performance table
    print("\n[INFO] Performance table...")
    perf_df = build_performance_table(all_results)
    perf_df.to_csv(os.path.join(OUT_DIR, "table_performance.csv"), index=False)
    print(f"  Saved table_performance.csv")

    print("\n[INFO] Modeling process reports...")
    fp_df, vif_df, ph_df, age_df = build_modeling_report_tables(all_results)
    fp_df.to_csv(os.path.join(OUT_DIR, "report_fp_selection.csv"), index=False)
    vif_df.to_csv(os.path.join(OUT_DIR, "report_vif_removed.csv"), index=False)
    ph_df.to_csv(os.path.join(OUT_DIR, "report_ph_assumption.csv"), index=False)
    age_df.to_csv(os.path.join(OUT_DIR, "report_age_interactions.csv"), index=False)
    print("  Saved report_fp_selection.csv")
    print("  Saved report_vif_removed.csv")
    print("  Saved report_ph_assumption.csv")
    print("  Saved report_age_interactions.csv")

    # Calibration plots
    print("\n[INFO] Calibration plot...")
    plot_calibration(all_results, OUT_DIR)

    # DCA plots
    print("\n[INFO] DCA plot...")
    plot_dca(all_results, OUT_DIR)

    print(f"\n{'='*70}")
    print(f"All outputs saved to: {OUT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
