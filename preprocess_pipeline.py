#!/usr/bin/env python3
"""
Preprocessing and imputation quality pipeline for T2D transition modeling.

Implements the user-specified workflow:
1) ID-level stratified Train/Val/Test split with rare-strata merging.
2) Variable filtering and structural-missing measured flags from table6.
3) Sex x age-group IQR clipping for continuous variables.
4) Wave-stratified imputation (MICE / missForest / median).
5) Imputation quality masking evaluation (table1.2, table1.3).
6) Downstream LightGBM evaluation for D0 and D2 (table1.4).
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import IterativeImputer
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split


RANDOM_SEED = 42
RARE_STRATA_MIN = 30


@dataclass
class VariableQualityRules:
    drop_vars: List[str]
    structural_missing_waves: Dict[str, List[int]]
    var_type_map: Dict[str, str]


@dataclass
class OutlierProfile:
    bounds_table: pd.DataFrame
    global_bounds: Dict[str, Tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="T2D preprocessing + imputation quality pipeline")
    parser.add_argument("--data", default="out/ana_dat_f.csv", help="Input row-level dataset path")
    parser.add_argument(
        "--quality-table",
        default="out/table6_variable_quality_by_wave.csv",
        help="Variable quality table path",
    )
    parser.add_argument("--out-dir", default="out/preprocess_v2", help="Output directory")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.10,
        help="Mask ratio for imputation quality evaluation on observed cells",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="GroupKFold splits")
    parser.add_argument("--cv-repeats", type=int, default=3, help="Repeated GroupKFold repeats")
    parser.add_argument(
        "--skip-lightgbm",
        action="store_true",
        help="Skip downstream LightGBM evaluation (table1.4)",
    )
    return parser.parse_args()


def safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def derive_wave(df: pd.DataFrame) -> pd.Series:
    wave = pd.Series(np.nan, index=df.index, dtype="float")

    if "wave_exam" in df.columns:
        wave = np.where(df["wave_exam"] == 2013, 1, wave)
        wave = np.where(df["wave_exam"] == 2018, 2, wave)
        wave = pd.Series(wave, index=df.index, dtype="float")

    start_year = safe_to_datetime(df["t_start"]).dt.year
    stop_year = safe_to_datetime(df["t_stop"]).dt.year

    wave = np.where(pd.isna(wave) & (start_year <= 2012), 0, wave)
    wave = np.where(pd.isna(wave) & (start_year >= 2013) & (start_year <= 2017), 1, wave)
    wave = np.where(pd.isna(wave) & (start_year >= 2018), 2, wave)

    wave = np.where(pd.isna(wave) & (stop_year <= 2012), 0, wave)
    wave = np.where(pd.isna(wave) & (stop_year >= 2013) & (stop_year <= 2017), 1, wave)
    wave = np.where(pd.isna(wave) & (stop_year >= 2018), 2, wave)

    wave = pd.Series(wave, index=df.index, dtype="float")

    if wave.isna().any():
        temp = pd.DataFrame(
            {
                "id": df["id"],
                "t_start_dt": safe_to_datetime(df["t_start"]),
                "t_stop_dt": safe_to_datetime(df["t_stop"]),
            },
            index=df.index,
        )
        temp_sorted = temp.sort_values(["id", "t_start_dt", "t_stop_dt"], kind="mergesort")
        order = temp_sorted.groupby("id").cumcount().clip(upper=2)
        fallback_rank = pd.Series(order.values, index=temp_sorted.index).reindex(df.index)
        wave = wave.fillna(fallback_rank.astype(float))

    wave = wave.fillna(0).astype(int)
    return wave


def build_id_summary(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["t_start_dt"] = safe_to_datetime(temp["t_start"])
    temp["t_stop_dt"] = safe_to_datetime(temp["t_stop"])
    temp = temp.sort_values(["id", "t_start_dt", "t_stop_dt"], kind="mergesort")

    grouped = temp.groupby("id", sort=False)

    entry_state = grouped["state_start"].first().astype("Int64")
    entry_wave = grouped["wave"].first().astype("Int64")

    ever_dm = (
        ((temp["state_start"] == 1) | (temp["state_stop"] == 1))
        .groupby(temp["id"])
        .max()
        .astype("Int64")
    )
    ever_pre = (
        ((temp["state_start"] == 2) | (temp["state_stop"] == 2))
        .groupby(temp["id"])
        .max()
        .astype("Int64")
    )

    summary = pd.DataFrame(
        {
            "id": entry_state.index,
            "entry_state": entry_state.values,
            "ever_dm": ever_dm.values,
            "ever_pre": ever_pre.values,
            "entry_wave": entry_wave.values,
        }
    )
    return summary


def compose_strata(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    out["s0"] = (
        "es"
        + out["entry_state"].astype(str)
        + "_dm"
        + out["ever_dm"].astype(str)
        + "_pre"
        + out["ever_pre"].astype(str)
        + "_w"
        + out["entry_wave"].astype(str)
    )
    out["s1"] = (
        "es"
        + out["entry_state"].astype(str)
        + "_dm"
        + out["ever_dm"].astype(str)
        + "_pre"
        + out["ever_pre"].astype(str)
        + "_wall"
    )
    out["s2"] = (
        "es"
        + out["entry_state"].astype(str)
        + "_dm"
        + out["ever_dm"].astype(str)
        + "_preall_wall"
    )
    out["s3"] = "es" + out["entry_state"].astype(str) + "_dmall_preall_wall"
    out["s4"] = "all"
    return out


def merge_rare_strata(summary: pd.DataFrame, min_count: int = RARE_STRATA_MIN) -> pd.DataFrame:
    out = compose_strata(summary)
    counts = {level: out[level].value_counts() for level in ["s0", "s1", "s2", "s3", "s4"]}

    merged: List[str] = []
    for _, row in out.iterrows():
        chosen = row["s4"]
        for level in ["s0", "s1", "s2", "s3", "s4"]:
            c = int(counts[level].get(row[level], 0))
            if c >= min_count:
                chosen = row[level]
                break
        merged.append(chosen)
    out["strata_merged"] = merged
    return out


def _safe_split(
    ids: np.ndarray,
    stratify: Optional[pd.Series],
    test_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if stratify is None or stratify.nunique() < 2:
        return train_test_split(ids, test_size=test_size, random_state=seed, shuffle=True)
    try:
        return train_test_split(ids, test_size=test_size, random_state=seed, shuffle=True, stratify=stratify)
    except ValueError:
        # Fallback if some strata remain too small for exact split.
        return train_test_split(ids, test_size=test_size, random_state=seed, shuffle=True)


def split_train_val_test(id_strata: pd.DataFrame, seed: int) -> pd.DataFrame:
    ids = id_strata["id"].to_numpy()
    strat = id_strata["strata_merged"]

    train_ids, temp_ids = _safe_split(ids, strat, test_size=0.20, seed=seed)

    temp_mask = id_strata["id"].isin(temp_ids)
    temp_df = id_strata.loc[temp_mask, ["id", "strata_merged"]]
    val_ids, test_ids = _safe_split(
        temp_df["id"].to_numpy(),
        temp_df["strata_merged"],
        test_size=0.50,
        seed=seed,
    )

    split_map = pd.DataFrame({"id": ids})
    split_map["split"] = "train"
    split_map.loc[split_map["id"].isin(val_ids), "split"] = "val"
    split_map.loc[split_map["id"].isin(test_ids), "split"] = "test"
    return split_map


def split_outcome_stats(summary: pd.DataFrame, split_map: pd.DataFrame) -> pd.DataFrame:
    merged = summary.merge(split_map, on="id", how="left")
    rows = []
    for sp in ["train", "val", "test"]:
        g = merged[merged["split"] == sp]
        rows.append(
            {
                "split": sp,
                "n_id": int(g["id"].nunique()),
                "ever_dm_n": int(g["ever_dm"].sum()),
                "ever_pre_n": int(g["ever_pre"].sum()),
                "ever_dm_rate": float(g["ever_dm"].mean()) if len(g) else np.nan,
                "ever_pre_rate": float(g["ever_pre"].mean()) if len(g) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def parse_rate(x: Any) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("%", "")
    if s in {"", "—", "-", "NA", "NaN", "nan"}:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_variable_quality_rules(quality_df: pd.DataFrame) -> VariableQualityRules:
    miss_cols = [c for c in quality_df.columns if "缺失率" in c]

    wave_col_map = {}
    for c in miss_cols:
        if "2008" in c:
            wave_col_map[c] = 0
        elif "2013" in c:
            wave_col_map[c] = 1
        elif "2018" in c:
            wave_col_map[c] = 2

    drop_vars: List[str] = []
    structural_missing_waves: Dict[str, List[int]] = {}
    var_type_map = {
        str(row["变量名"]): str(row["变量类型"]) for _, row in quality_df[["变量名", "变量类型"]].iterrows()
    }

    for _, row in quality_df.iterrows():
        var = str(row["变量名"])
        rates = {wave_col_map[c]: parse_rate(row[c]) for c in miss_cols if c in wave_col_map}
        waves_ge95 = [w for w, r in rates.items() if pd.notna(r) and r >= 95.0]
        any_lt40 = any(pd.notna(r) and r < 40.0 for r in rates.values())

        if len(waves_ge95) >= 2:
            drop_vars.append(var)
        elif len(waves_ge95) >= 1 and any_lt40:
            structural_missing_waves[var] = sorted(waves_ge95)

    return VariableQualityRules(
        drop_vars=sorted(set(drop_vars)),
        structural_missing_waves=structural_missing_waves,
        var_type_map=var_type_map,
    )


def apply_variable_rules(
    df: pd.DataFrame,
    rules: VariableQualityRules,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    out = df.copy()
    drop_existing = [c for c in rules.drop_vars if c in out.columns]
    if drop_existing:
        out = out.drop(columns=drop_existing)

    flag_map: Dict[str, str] = {}
    for var, waves in rules.structural_missing_waves.items():
        if var not in out.columns:
            continue
        flag_col = f"measured_flag_{var}"
        out[flag_col] = (~out["wave"].isin(waves)).astype("int8")
        flag_map[var] = flag_col

    return out, flag_map


def to_numeric_inplace(df: pd.DataFrame, cols: Sequence[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


def build_age_group(age: pd.Series) -> pd.Series:
    bins = [-np.inf, 49, 59, 69, np.inf]
    labels = ["<=49", "50-59", "60-69", ">=70"]
    return pd.cut(age, bins=bins, labels=labels)


def fit_outlier_profile(
    train_df: pd.DataFrame,
    continuous_vars: Sequence[str],
    min_group_n: int = 20,
) -> OutlierProfile:
    train = train_df.copy()
    train["age_group"] = build_age_group(train["age"])

    rows = []
    global_bounds: Dict[str, Tuple[float, float]] = {}

    for var in continuous_vars:
        if var not in train.columns:
            continue

        vals_all = train[var].dropna()
        if vals_all.empty:
            continue

        q1_all = vals_all.quantile(0.25)
        q3_all = vals_all.quantile(0.75)
        iqr_all = q3_all - q1_all
        lower_all = q1_all - 1.5 * iqr_all
        upper_all = q3_all + 1.5 * iqr_all
        global_bounds[var] = (float(lower_all), float(upper_all))

        rows.append(
            {
                "variable": var,
                "gender": "ALL",
                "age_group": "ALL",
                "lower": float(lower_all),
                "upper": float(upper_all),
                "outlier_ratio": float(((vals_all < lower_all) | (vals_all > upper_all)).mean()),
                "n_non_missing": int(vals_all.shape[0]),
            }
        )

        g = train.groupby(["gender", "age_group"], dropna=False, observed=False)
        for (gender, age_group), sub in g:
            vals = sub[var].dropna()
            if vals.shape[0] < min_group_n:
                lower, upper = lower_all, upper_all
                ratio = float(((vals < lower) | (vals > upper)).mean()) if vals.shape[0] else np.nan
            else:
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                ratio = float(((vals < lower) | (vals > upper)).mean())

            rows.append(
                {
                    "variable": var,
                    "gender": gender,
                    "age_group": str(age_group),
                    "lower": float(lower),
                    "upper": float(upper),
                    "outlier_ratio": ratio,
                    "n_non_missing": int(vals.shape[0]),
                }
            )

    bounds_table = pd.DataFrame(rows)
    return OutlierProfile(bounds_table=bounds_table, global_bounds=global_bounds)


def clip_outliers_with_profile(
    df: pd.DataFrame,
    profile: OutlierProfile,
    continuous_vars: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    out["age_group"] = build_age_group(out["age"])

    for var in continuous_vars:
        if var not in out.columns or var not in profile.global_bounds:
            continue

        sub = profile.bounds_table[
            (profile.bounds_table["variable"] == var)
            & (profile.bounds_table["gender"] != "ALL")
            & (profile.bounds_table["age_group"] != "ALL")
        ][["gender", "age_group", "lower", "upper"]]

        merged = out[["gender", "age_group"]].merge(sub, on=["gender", "age_group"], how="left")

        lower_all, upper_all = profile.global_bounds[var]
        lower = merged["lower"].fillna(lower_all).to_numpy()
        upper = merged["upper"].fillna(upper_all).to_numpy()
        values = out[var].to_numpy(dtype=float)

        clipped = np.where(np.isnan(values), values, np.minimum(np.maximum(values, lower), upper))
        out[var] = clipped

    out = out.drop(columns=["age_group"])
    return out


def get_var_lists(
    df: pd.DataFrame,
    rules: VariableQualityRules,
) -> Tuple[List[str], List[str]]:
    continuous = [
        var
        for var, tp in rules.var_type_map.items()
        if ("连续" in tp) and (var in df.columns)
    ]
    categorical = [
        var
        for var, tp in rules.var_type_map.items()
        if ("分" in tp) and (var in df.columns)
    ]
    return sorted(continuous), sorted(categorical)


def format_mean_sd(values: Iterable[float], digits: int = 4) -> str:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return "nan±nan"
    return f"{arr.mean():.{digits}f}±{arr.std(ddof=1):.{digits}f}" if arr.size > 1 else f"{arr.mean():.{digits}f}±0"


def _fit_iterative(
    X: pd.DataFrame,
    method: str,
    random_state: int,
) -> Any:
    if method == "mice":
        imp = IterativeImputer(random_state=random_state, max_iter=10, sample_posterior=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            imp.fit(X)
        return imp
    if method == "missforest":
        estimator = RandomForestRegressor(
            n_estimators=150,
            random_state=random_state,
            n_jobs=-1,
        )
        imp = IterativeImputer(
            estimator=estimator,
            random_state=random_state,
            max_iter=10,
            sample_posterior=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            imp.fit(X)
        return imp
    raise ValueError(f"Unsupported iterative method: {method}")


def _fit_miceforest(X: pd.DataFrame, random_state: int) -> Any:
    import miceforest as mf  # type: ignore

    X_reset = X.reset_index(drop=True)
    kernel = mf.ImputationKernel(
        data=X_reset,
        num_datasets=1,
        save_all_iterations_data=True,
        random_state=random_state,
    )
    kernel.mice(iterations=3, verbose=False)
    return kernel


def _transform_miceforest(kernel: Any, X: pd.DataFrame) -> pd.DataFrame:
    # Compatible with common miceforest variants. miceforest requires reset index.
    orig_index = X.index
    X_reset = X.reset_index(drop=True)
    try:
        new_data = kernel.impute_new_data(new_data=X_reset, datasets=[0])
        out = new_data.complete_data(dataset=0)
    except Exception:
        new_data = kernel.impute_new_data(X_reset, datasets=[0])
        out = new_data.complete_data(dataset=0)
    out.index = orig_index
    return out


def _fit_missforest_pkg(X: pd.DataFrame) -> Any:
    # Package name can expose either `missforest` or `MissForest`.
    try:
        from missforest import MissForest  # type: ignore
    except Exception:
        from MissForest import MissForest  # type: ignore

    imputer = MissForest(verbose=0)
    imputer.fit(X)
    return imputer


def _transform_missforest_pkg(imputer: Any, X: pd.DataFrame) -> pd.DataFrame:
    out = imputer.transform(X)
    if isinstance(out, pd.DataFrame):
        return out
    return pd.DataFrame(out, index=X.index, columns=X.columns)


def fit_wave_imputer(
    train_df: pd.DataFrame,
    method: str,
    cont_cols: Sequence[str],
    cat_cols: Sequence[str],
    random_state: int,
    use_cat_mode: bool = False,
) -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "method": method,
        "cont_cols": list(cont_cols),
        "cat_cols": list(cat_cols),
        "models": {},
        "fallback": {},
    }

    all_waves = sorted(pd.Series(train_df["wave"].dropna().unique()).astype(int).tolist())

    if method == "median":
        for wave in all_waves + ["global"]:
            sub = train_df if wave == "global" else train_df[train_df["wave"] == wave]
            cont_fill = {c: sub[c].median(skipna=True) for c in cont_cols if c in sub.columns}
            cat_fill = {}
            if use_cat_mode:
                for c in cat_cols:
                    if c in sub.columns:
                        mode = sub[c].mode(dropna=True)
                        cat_fill[c] = mode.iloc[0] if not mode.empty else np.nan
            state["models"][wave] = {"cont_fill": cont_fill, "cat_fill": cat_fill}
        return state

    if method == "mice":
        try:
            import miceforest  # noqa: F401

            backend = "miceforest"
        except Exception:
            backend = "iterative"
        state["backend"] = backend
    elif method == "missforest":
        try:
            from missforest import MissForest as _MF  # noqa: F401

            backend = "missforest_pkg"
        except Exception:
            try:
                import missforest as _MF2  # noqa: F401

                backend = "missforest_pkg"
            except Exception:
                backend = "iterative_rf"
        state["backend"] = backend
    else:
        raise ValueError(f"Unknown method: {method}")

    def fit_one(sub: pd.DataFrame) -> Dict[str, Any]:
        use_cols = [c for c in cont_cols if c in sub.columns and sub[c].notna().any()]
        if not use_cols:
            return {"use_cols": [], "model": None}

        X = sub[use_cols].copy()

        if state["backend"] == "miceforest":
            try:
                model = _fit_miceforest(X, random_state=random_state)
                return {"use_cols": use_cols, "model": model}
            except Exception as e:
                warnings.warn(f"miceforest backend failed, fallback to iterative: {e}")
                state["backend"] = "iterative"

        if state["backend"] == "missforest_pkg":
            try:
                model = _fit_missforest_pkg(X)
                return {"use_cols": use_cols, "model": model}
            except Exception as e:
                warnings.warn(f"MissForest package backend failed, fallback to iterative_rf: {e}")
                state["backend"] = "iterative_rf"

        if state["backend"] == "iterative":
            model = _fit_iterative(X, method="mice", random_state=random_state)
            return {"use_cols": use_cols, "model": model}

        if state["backend"] == "iterative_rf":
            model = _fit_iterative(X, method="missforest", random_state=random_state)
            return {"use_cols": use_cols, "model": model}

        raise RuntimeError(f"Unsupported backend {state['backend']}")

    for wave in all_waves:
        sub = train_df[train_df["wave"] == wave]
        state["models"][wave] = fit_one(sub)

    state["models"]["global"] = fit_one(train_df)
    return state


def transform_wave_imputer(df: pd.DataFrame, state: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    method = state["method"]
    cont_cols: List[str] = state["cont_cols"]
    cat_cols: List[str] = state["cat_cols"]

    if method == "median":
        for wave in sorted(pd.Series(out["wave"].dropna().unique()).astype(int).tolist()):
            idx = out.index[out["wave"] == wave]
            model = state["models"].get(wave, state["models"]["global"])
            for c in cont_cols:
                if c in out.columns:
                    out.loc[idx, c] = out.loc[idx, c].fillna(model["cont_fill"].get(c, np.nan))
            for c in cat_cols:
                if c in out.columns:
                    out.loc[idx, c] = out.loc[idx, c].fillna(model["cat_fill"].get(c, np.nan))
        return out

    backend = state.get("backend", "iterative")

    def transform_block(idx: pd.Index, model_info: Dict[str, Any]) -> None:
        use_cols = model_info.get("use_cols", [])
        model = model_info.get("model", None)
        if not use_cols or model is None or len(idx) == 0:
            return

        X = out.loc[idx, use_cols].copy()

        if backend == "miceforest":
            try:
                X_imp = _transform_miceforest(model, X)
            except Exception as e:
                warnings.warn(
                    f"miceforest transform failed for block; keep original values. reason={e}"
                )
                X_imp = X
        elif backend == "missforest_pkg":
            X_imp = _transform_missforest_pkg(model, X)
        else:
            X_imp = pd.DataFrame(model.transform(X), index=X.index, columns=use_cols)

        for c in use_cols:
            out.loc[idx, c] = X_imp[c]

    for wave in sorted(pd.Series(out["wave"].dropna().unique()).astype(int).tolist()):
        idx = out.index[out["wave"] == wave]
        model_info = state["models"].get(wave, state["models"]["global"])
        transform_block(idx, model_info)

    return out


def build_imputed_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    method: str,
    cont_cols: Sequence[str],
    cat_cols: Sequence[str],
    seed: int,
    use_cat_mode: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    state = fit_wave_imputer(
        train_df=train_df,
        method=method,
        cont_cols=cont_cols,
        cat_cols=cat_cols,
        random_state=seed,
        use_cat_mode=use_cat_mode,
    )
    train_i = transform_wave_imputer(train_df, state)
    val_i = transform_wave_imputer(val_df, state)
    test_i = transform_wave_imputer(test_df, state)
    return train_i, val_i, test_i, state


def mask_cells_by_wave(
    train_df: pd.DataFrame,
    cont_cols: Sequence[str],
    mask_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    masked = train_df.copy()
    records: List[Dict[str, Any]] = []

    for wave in sorted(pd.Series(train_df["wave"].dropna().unique()).astype(int).tolist()):
        idx_wave = train_df.index[train_df["wave"] == wave]
        for c in cont_cols:
            if c not in train_df.columns:
                continue
            observed_idx = idx_wave[train_df.loc[idx_wave, c].notna().to_numpy()]
            n_obs = len(observed_idx)
            if n_obs == 0:
                continue
            n_mask = max(1, int(math.floor(n_obs * mask_ratio)))
            pick = rng.choice(observed_idx.to_numpy(), size=min(n_mask, n_obs), replace=False)
            truth_vals = train_df.loc[pick, c].to_numpy()
            masked.loc[pick, c] = np.nan
            for i, v in zip(pick.tolist(), truth_vals.tolist()):
                records.append({"index": i, "wave": wave, "variable": c, "truth": float(v)})

    truth = pd.DataFrame(records)
    return masked, truth


def eval_imputation_quality(
    train_df: pd.DataFrame,
    cont_cols: Sequence[str],
    methods: Sequence[str],
    mask_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    masked, truth = mask_cells_by_wave(train_df, cont_cols, mask_ratio=mask_ratio, seed=seed)

    table12_rows: List[Dict[str, Any]] = []
    table13_rows: List[Dict[str, Any]] = []

    for method in methods:
        use_cat_mode = False
        imputed, _, _, _ = build_imputed_splits(
            train_df=masked,
            val_df=masked.iloc[:0].copy(),
            test_df=masked.iloc[:0].copy(),
            method=method,
            cont_cols=cont_cols,
            cat_cols=[],
            seed=seed,
            use_cat_mode=use_cat_mode,
        )

        joined = truth.copy()
        joined["pred"] = [imputed.loc[int(i), var] for i, var in zip(joined["index"], joined["variable"])]
        joined["abs_err"] = np.abs(joined["pred"] - joined["truth"])
        joined["sq_err"] = (joined["pred"] - joined["truth"]) ** 2

        for (wave, var), sub in joined.groupby(["wave", "variable"]):
            if len(sub) == 0:
                continue
            rmse = math.sqrt(float(sub["sq_err"].mean()))
            mae = float(sub["abs_err"].mean())
            p90ae = float(np.nanquantile(sub["abs_err"], 0.90))
            table12_rows.append(
                {
                    "method": method,
                    "wave": int(wave),
                    "variable": var,
                    "n_masked": int(len(sub)),
                    "rmse": rmse,
                    "mae": mae,
                    "p90_abs_error": p90ae,
                }
            )

        sub_m = pd.DataFrame([r for r in table12_rows if r["method"] == method])
        table13_rows.append(
            {
                "method": method,
                "rmse_mean_sd": format_mean_sd(sub_m["rmse"].tolist(), digits=4),
                "mae_mean_sd": format_mean_sd(sub_m["mae"].tolist(), digits=4),
                "p90ae_mean_sd": format_mean_sd(sub_m["p90_abs_error"].tolist(), digits=4),
                "n_variable_wave_cells": int(len(sub_m)),
            }
        )

    table12 = pd.DataFrame(table12_rows).sort_values(["method", "wave", "variable"])
    table13 = pd.DataFrame(table13_rows).sort_values(["method"])
    return table12, table13


def multiclass_brier(y_true: np.ndarray, proba: np.ndarray, n_classes: int) -> float:
    y_onehot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_onehot - proba) ** 2, axis=1)))


def multiclass_concordance(
    y_true_label: np.ndarray,
    proba: np.ndarray,
    class_order: Sequence[int],
) -> float:
    # Ordinalized risk score for pairwise concordance.
    risk_map = {0: 0.0, 2: 1.0, 1: 2.0}
    class_risk = np.array([risk_map.get(c, float(i)) for i, c in enumerate(class_order)], dtype=float)
    risk_score = proba.dot(class_risk)
    y_ord = np.array([risk_map.get(int(y), np.nan) for y in y_true_label], dtype=float)

    n = len(y_ord)
    concordant = 0.0
    comparable = 0
    for i in range(n):
        yi = y_ord[i]
        if np.isnan(yi):
            continue
        for j in range(i + 1, n):
            yj = y_ord[j]
            if np.isnan(yj) or yi == yj:
                continue
            comparable += 1
            diff_truth = yi - yj
            diff_score = risk_score[i] - risk_score[j]
            if diff_score == 0:
                concordant += 0.5
            elif diff_truth * diff_score > 0:
                concordant += 1.0
    if comparable == 0:
        return np.nan
    return float(concordant / comparable)


def compute_metrics(
    y_true_enc: np.ndarray,
    y_true_label: np.ndarray,
    pred_label_enc: np.ndarray,
    proba: np.ndarray,
    class_order: Sequence[int],
    null_prob: np.ndarray,
) -> Dict[str, float]:
    n_classes = len(class_order)

    macro_f1 = float(f1_score(y_true_enc, pred_label_enc, average="macro"))

    try:
        auc = float(roc_auc_score(y_true_enc, proba, average="macro", multi_class="ovr"))
    except Exception:
        auc = np.nan

    brier = multiclass_brier(y_true_enc, proba, n_classes=n_classes)

    null_proba = np.repeat(null_prob.reshape(1, -1), repeats=len(y_true_enc), axis=0)
    brier_null = multiclass_brier(y_true_enc, null_proba, n_classes=n_classes)
    r2 = np.nan if brier_null <= 0 else float(1.0 - (brier / brier_null))

    c_stat = multiclass_concordance(y_true_label, proba, class_order=class_order)
    d_stat = np.nan if np.isnan(c_stat) else float(2.0 * c_stat - 1.0)

    return {
        "macro_f1": macro_f1,
        "ovr_auc": auc,
        "brier": brier,
        "d_statistic": d_stat,
        "r2": r2,
        "harrell_c": c_stat,
    }


def _shuffle_for_repeat(
    df: pd.DataFrame,
    seed: int,
    repeat_idx: int,
) -> pd.DataFrame:
    return df.sample(frac=1.0, random_state=seed + repeat_idx).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    drop_cols = {"id", "t_start", "t_stop", "time", "split", "state_stop"}
    return [c for c in df.columns if c not in drop_cols]


def encode_target(y: pd.Series, class_order: Sequence[int]) -> pd.Series:
    mapper = {c: i for i, c in enumerate(class_order)}
    return y.map(mapper)


def evaluate_downstream_for_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_name: str,
    class_order: Sequence[int],
    strategies: Sequence[str],
    impute_cont_cols: Sequence[str],
    cat_cols: Sequence[str],
    seed: int,
    cv_splits: int,
    cv_repeats: int,
) -> pd.DataFrame:
    try:
        from lightgbm import LGBMClassifier
    except Exception as e:
        raise ImportError(
            "lightgbm is required for table1.4. Install dependencies from requirements_imputation.txt"
        ) from e

    rows = []

    for strategy in strategies:
        cv_metrics: List[Dict[str, float]] = []

        for repeat in range(cv_repeats):
            shuffled = _shuffle_for_repeat(train_df, seed=seed, repeat_idx=repeat)
            gkf = GroupKFold(n_splits=cv_splits)

            for fold_idx, (tr_idx, va_idx) in enumerate(
                gkf.split(shuffled, groups=shuffled["id"]), start=1
            ):
                fold_train = shuffled.iloc[tr_idx].copy()
                fold_val = shuffled.iloc[va_idx].copy()

                if strategy == "no_impute":
                    tr_i, va_i = fold_train, fold_val
                elif strategy == "mice":
                    tr_i, va_i, _, _ = build_imputed_splits(
                        train_df=fold_train,
                        val_df=fold_val,
                        test_df=fold_val.iloc[:0].copy(),
                        method="mice",
                        cont_cols=impute_cont_cols,
                        cat_cols=[],
                        seed=seed + repeat + fold_idx,
                        use_cat_mode=False,
                    )
                elif strategy == "missforest":
                    tr_i, va_i, _, _ = build_imputed_splits(
                        train_df=fold_train,
                        val_df=fold_val,
                        test_df=fold_val.iloc[:0].copy(),
                        method="missforest",
                        cont_cols=impute_cont_cols,
                        cat_cols=[],
                        seed=seed + repeat + fold_idx,
                        use_cat_mode=False,
                    )
                elif strategy == "median_mode":
                    tr_i, va_i, _, _ = build_imputed_splits(
                        train_df=fold_train,
                        val_df=fold_val,
                        test_df=fold_val.iloc[:0].copy(),
                        method="median",
                        cont_cols=impute_cont_cols,
                        cat_cols=cat_cols,
                        seed=seed + repeat + fold_idx,
                        use_cat_mode=True,
                    )
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                feature_cols = get_feature_columns(tr_i)
                X_tr = tr_i[feature_cols]
                X_va = va_i[feature_cols]

                y_tr_label = tr_i["state_stop"].astype(int)
                y_va_label = va_i["state_stop"].astype(int)

                y_tr = encode_target(y_tr_label, class_order=class_order)
                y_va = encode_target(y_va_label, class_order=class_order)

                valid_train = y_tr.notna()
                valid_val = y_va.notna()
                X_tr, y_tr, y_tr_label = X_tr.loc[valid_train], y_tr.loc[valid_train], y_tr_label.loc[valid_train]
                X_va, y_va, y_va_label = X_va.loc[valid_val], y_va.loc[valid_val], y_va_label.loc[valid_val]

                if y_tr.nunique() < 2 or len(X_va) == 0:
                    continue

                clf = LGBMClassifier(
                    objective="multiclass",
                    num_class=len(class_order),
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=seed + repeat + fold_idx,
                    n_jobs=-1,
                    verbose=-1,
                )
                clf.fit(X_tr, y_tr.astype(int))

                proba = clf.predict_proba(X_va)
                pred = np.argmax(proba, axis=1)

                null_prob = (
                    y_tr.value_counts(normalize=True)
                    .reindex(range(len(class_order)), fill_value=0.0)
                    .to_numpy(dtype=float)
                )

                met = compute_metrics(
                    y_true_enc=y_va.astype(int).to_numpy(),
                    y_true_label=y_va_label.astype(int).to_numpy(),
                    pred_label_enc=pred.astype(int),
                    proba=np.asarray(proba, dtype=float),
                    class_order=class_order,
                    null_prob=null_prob,
                )
                cv_metrics.append(met)

        if len(cv_metrics) == 0:
            cv_summary = {
                "macro_f1_cv": "nan±nan",
                "ovr_auc_cv": "nan±nan",
                "brier_cv": "nan±nan",
                "d_statistic_cv": "nan±nan",
                "r2_cv": "nan±nan",
                "harrell_c_cv": "nan±nan",
            }
        else:
            cv_df = pd.DataFrame(cv_metrics)
            cv_summary = {
                "macro_f1_cv": format_mean_sd(cv_df["macro_f1"], digits=4),
                "ovr_auc_cv": format_mean_sd(cv_df["ovr_auc"], digits=4),
                "brier_cv": format_mean_sd(cv_df["brier"], digits=4),
                "d_statistic_cv": format_mean_sd(cv_df["d_statistic"], digits=4),
                "r2_cv": format_mean_sd(cv_df["r2"], digits=4),
                "harrell_c_cv": format_mean_sd(cv_df["harrell_c"], digits=4),
            }

        # Final train -> test evaluation
        if strategy == "no_impute":
            tr_f, te_f = train_df.copy(), test_df.copy()
        elif strategy == "mice":
            tr_f, _, te_f, _ = build_imputed_splits(
                train_df=train_df,
                val_df=train_df.iloc[:0].copy(),
                test_df=test_df,
                method="mice",
                cont_cols=impute_cont_cols,
                cat_cols=[],
                seed=seed,
                use_cat_mode=False,
            )
        elif strategy == "missforest":
            tr_f, _, te_f, _ = build_imputed_splits(
                train_df=train_df,
                val_df=train_df.iloc[:0].copy(),
                test_df=test_df,
                method="missforest",
                cont_cols=impute_cont_cols,
                cat_cols=[],
                seed=seed,
                use_cat_mode=False,
            )
        elif strategy == "median_mode":
            tr_f, _, te_f, _ = build_imputed_splits(
                train_df=train_df,
                val_df=train_df.iloc[:0].copy(),
                test_df=test_df,
                method="median",
                cont_cols=impute_cont_cols,
                cat_cols=cat_cols,
                seed=seed,
                use_cat_mode=True,
            )
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        feature_cols = get_feature_columns(tr_f)
        X_tr = tr_f[feature_cols]
        X_te = te_f[feature_cols]

        y_tr_label = tr_f["state_stop"].astype(int)
        y_te_label = te_f["state_stop"].astype(int)
        y_tr = encode_target(y_tr_label, class_order=class_order)
        y_te = encode_target(y_te_label, class_order=class_order)

        valid_train = y_tr.notna()
        valid_test = y_te.notna()
        X_tr, y_tr, y_tr_label = X_tr.loc[valid_train], y_tr.loc[valid_train], y_tr_label.loc[valid_train]
        X_te, y_te, y_te_label = X_te.loc[valid_test], y_te.loc[valid_test], y_te_label.loc[valid_test]

        if y_tr.nunique() < 2 or len(X_te) == 0:
            test_metrics = {k: np.nan for k in ["macro_f1", "ovr_auc", "brier", "d_statistic", "r2", "harrell_c"]}
        else:
            clf = LGBMClassifier(
                objective="multiclass",
                num_class=len(class_order),
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=seed,
                n_jobs=-1,
                verbose=-1,
            )
            clf.fit(X_tr, y_tr.astype(int))
            proba = clf.predict_proba(X_te)
            pred = np.argmax(proba, axis=1)

            null_prob = (
                y_tr.value_counts(normalize=True)
                .reindex(range(len(class_order)), fill_value=0.0)
                .to_numpy(dtype=float)
            )
            test_metrics = compute_metrics(
                y_true_enc=y_te.astype(int).to_numpy(),
                y_true_label=y_te_label.astype(int).to_numpy(),
                pred_label_enc=pred.astype(int),
                proba=np.asarray(proba, dtype=float),
                class_order=class_order,
                null_prob=null_prob,
            )

        rows.append(
            {
                "dataset": dataset_name,
                "imputation": strategy,
                "macro-F1 (CV mean±sd)": cv_summary["macro_f1_cv"],
                "OVR-AUC (CV mean±sd)": cv_summary["ovr_auc_cv"],
                "Brier (CV mean±sd)": cv_summary["brier_cv"],
                "D-statistic (CV mean±sd)": cv_summary["d_statistic_cv"],
                "R² (CV mean±sd)": cv_summary["r2_cv"],
                "Harrell's C-statistic (CV mean±sd)": cv_summary["harrell_c_cv"],
                "Test macro-F1": test_metrics["macro_f1"],
                "Test OVR-AUC": test_metrics["ovr_auc"],
                "Test Brier": test_metrics["brier"],
                "Test D-statistic": test_metrics["d_statistic"],
                "Test R²": test_metrics["r2"],
                "Test Harrell's C-statistic": test_metrics["harrell_c"],
            }
        )

    return pd.DataFrame(rows)


def save_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False)
    elif path.suffix.lower() in {".pkl", ".pickle"}:
        df.to_pickle(path)
    else:
        df.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 0: load
    df = pd.read_csv(args.data)
    quality_df = pd.read_csv(args.quality_table)

    original_columns = df.columns.tolist()
    first_17_cols = original_columns[:17]

    # Step 1: derive wave, build strata, split
    df["wave"] = derive_wave(df)
    id_summary = build_id_summary(df)
    strata_df = merge_rare_strata(id_summary, min_count=RARE_STRATA_MIN)
    split_map = split_train_val_test(strata_df[["id", "strata_merged"]], seed=args.seed)
    split_stats = split_outcome_stats(id_summary, split_map)

    df = df.merge(split_map, on="id", how="left")

    save_df(id_summary, out_dir / "id_summary_labels.csv")
    save_df(strata_df[["id", "s0", "s1", "s2", "s3", "strata_merged"]], out_dir / "id_strata_merge_map.csv")
    save_df(split_map, out_dir / "id_split_map.csv")
    save_df(split_stats, out_dir / "split_ever_dm_ever_pre_stats.csv")
    save_df(df, out_dir / "step1_with_split.csv")

    # Step 2: variable quality rule processing
    rules = parse_variable_quality_rules(quality_df)
    df_step2, flag_map = apply_variable_rules(df, rules)

    continuous_vars, categorical_vars = get_var_lists(df_step2, rules)
    numeric_targets = sorted(set(continuous_vars + categorical_vars + ["age", "gender", "state_start", "state_stop"]))
    to_numeric_inplace(df_step2, [c for c in numeric_targets if c in df_step2.columns])

    save_df(df_step2, out_dir / "step2_structural_processed.csv")

    # Step 3: outlier clipping (fit on train only)
    train_df = df_step2[df_step2["split"] == "train"].copy()
    val_df = df_step2[df_step2["split"] == "val"].copy()
    test_df = df_step2[df_step2["split"] == "test"].copy()

    outlier_profile = fit_outlier_profile(train_df, continuous_vars=continuous_vars, min_group_n=20)
    train_clip = clip_outliers_with_profile(train_df, outlier_profile, continuous_vars=continuous_vars)
    val_clip = clip_outliers_with_profile(val_df, outlier_profile, continuous_vars=continuous_vars)
    test_clip = clip_outliers_with_profile(test_df, outlier_profile, continuous_vars=continuous_vars)

    save_df(outlier_profile.bounds_table, out_dir / "table1.1.csv")
    save_df(train_clip, out_dir / "step3_train_clipped.csv")
    save_df(val_clip, out_dir / "step3_val_clipped.csv")
    save_df(test_clip, out_dir / "step3_test_clipped.csv")

    # Step 4: imputation for continuous non-structural vars excluding first 17 columns
    structural_vars = set(rules.structural_missing_waves.keys())
    impute_cont_cols = [
        c
        for c in continuous_vars
        if (c not in first_17_cols) and (c not in structural_vars) and (c in train_clip.columns)
    ]
    impute_cat_cols = [c for c in categorical_vars if c in train_clip.columns and c not in first_17_cols]

    mice_train, mice_val, mice_test, mice_state = build_imputed_splits(
        train_df=train_clip,
        val_df=val_clip,
        test_df=test_clip,
        method="mice",
        cont_cols=impute_cont_cols,
        cat_cols=[],
        seed=args.seed,
        use_cat_mode=False,
    )
    missf_train, missf_val, missf_test, missf_state = build_imputed_splits(
        train_df=train_clip,
        val_df=val_clip,
        test_df=test_clip,
        method="missforest",
        cont_cols=impute_cont_cols,
        cat_cols=[],
        seed=args.seed,
        use_cat_mode=False,
    )
    med_train, med_val, med_test, med_state = build_imputed_splits(
        train_df=train_clip,
        val_df=val_clip,
        test_df=test_clip,
        method="median",
        cont_cols=impute_cont_cols,
        cat_cols=impute_cat_cols,
        seed=args.seed,
        use_cat_mode=True,
    )

    save_df(mice_train, out_dir / "step4_train_mice.csv")
    save_df(mice_val, out_dir / "step4_val_mice.csv")
    save_df(mice_test, out_dir / "step4_test_mice.csv")

    save_df(missf_train, out_dir / "step4_train_missforest.csv")
    save_df(missf_val, out_dir / "step4_val_missforest.csv")
    save_df(missf_test, out_dir / "step4_test_missforest.csv")

    save_df(med_train, out_dir / "step4_train_median_mode.csv")
    save_df(med_val, out_dir / "step4_val_median_mode.csv")
    save_df(med_test, out_dir / "step4_test_median_mode.csv")

    # Step 5: imputation quality evaluation (training set, wave-stratified masking)
    table12, table13 = eval_imputation_quality(
        train_df=train_clip,
        cont_cols=impute_cont_cols,
        methods=["mice", "missforest", "median"],
        mask_ratio=args.mask_ratio,
        seed=args.seed,
    )
    save_df(table12, out_dir / "table1.2.csv")
    save_df(table13, out_dir / "table1.3.csv")

    # Step 6: downstream LightGBM evaluation
    table14 = pd.DataFrame()
    if not args.skip_lightgbm:
        d0_train = train_clip[train_clip["state_start"] == 0].copy()
        d0_test = test_clip[test_clip["state_start"] == 0].copy()
        d2_train = train_clip[train_clip["state_start"] == 2].copy()
        d2_test = test_clip[test_clip["state_start"] == 2].copy()

        t14_d0 = evaluate_downstream_for_dataset(
            train_df=d0_train,
            test_df=d0_test,
            dataset_name="D0",
            class_order=[0, 2, 1],
            strategies=["no_impute", "mice", "missforest", "median_mode"],
            impute_cont_cols=impute_cont_cols,
            cat_cols=impute_cat_cols,
            seed=args.seed,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
        )
        t14_d2 = evaluate_downstream_for_dataset(
            train_df=d2_train,
            test_df=d2_test,
            dataset_name="D2",
            class_order=[2, 0, 1],
            strategies=["no_impute", "mice", "missforest", "median_mode"],
            impute_cont_cols=impute_cont_cols,
            cat_cols=impute_cat_cols,
            seed=args.seed,
            cv_splits=args.cv_splits,
            cv_repeats=args.cv_repeats,
        )
        table14 = pd.concat([t14_d0, t14_d2], axis=0, ignore_index=True)

    save_df(table14, out_dir / "table1.4.csv")

    metadata = {
        "data": args.data,
        "quality_table": args.quality_table,
        "out_dir": str(out_dir),
        "seed": args.seed,
        "mask_ratio": args.mask_ratio,
        "cv_splits": args.cv_splits,
        "cv_repeats": args.cv_repeats,
        "first_17_cols": first_17_cols,
        "dropped_vars_step2": rules.drop_vars,
        "structural_missing_vars_step2": sorted(rules.structural_missing_waves.keys()),
        "structural_flag_map": flag_map,
        "continuous_vars": continuous_vars,
        "categorical_vars": categorical_vars,
        "impute_cont_cols_step4": impute_cont_cols,
        "impute_cat_cols_step6_median_mode": impute_cat_cols,
        "mice_backend": mice_state.get("backend", "median"),
        "missforest_backend": missf_state.get("backend", "median"),
    }
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
