#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional, Union, Mapping

import logging
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


# In[ ]:


@dataclass(frozen=True)
class PreprocessedData:
    """Container holding all preprocessed dataframes."""
    temp: pd.DataFrame
    heart_rates: pd.DataFrame
    so2: pd.DataFrame
    bp: pd.DataFrame
    crp: pd.DataFrame
    bili: pd.DataFrame
    leua: pd.DataFrame
    krea: pd.DataFrame
    hb: pd.DataFrame
    ab_groups: pd.DataFrame


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _add_recorded_date(
    df: pd.DataFrame,
    *,
    time_column: str = "recorded_time",
    out_column: str = "recorded_date",
    utc: bool = True,
) -> pd.DataFrame:
    """Parse timestamps and add a date column derived from recorded_time."""
    _require_columns(df, [time_column], name="dataframe")
    dt = pd.to_datetime(df[time_column], utc=utc, errors="coerce")
    df = df.copy()
    df[time_column] = dt
    df[out_column] = dt.dt.date
    return df


def _rename_value_column(df: pd.DataFrame, new_name: str) -> pd.DataFrame:
    """Rename generic 'value' column to a metric-specific name."""
    _require_columns(df, ["value"], name=f"df for {new_name}")
    return df.rename(columns={"value": new_name})


def _dropna_in_value(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in `value` column (or metric column if already renamed)."""
    # prefer 'value' if present, else dropna over all columns
    if "value" in df.columns:
        return df.dropna(subset=["value"])
    return df.dropna()


def preprocess_from_dataframes(
    df_temp: pd.DataFrame,
    df_heart_rates: pd.DataFrame,
    df_so2: pd.DataFrame,
    df_bp: pd.DataFrame,
    df_crp: pd.DataFrame,
    df_bili: pd.DataFrame,
    df_leua: pd.DataFrame,
    df_krea: pd.DataFrame,
    df_hb: pd.DataFrame,
    df_ab_groups: pd.DataFrame,
    *,
    strict: bool = True,
    copy: bool = True,
) -> PreprocessedData:
    """
    Preprocess measurements data.

    Steps performed:
    - Drop NaNs for selected lab values
    - Parse `recorded_time` as UTC timestamps and add `recorded_date`
    - Create `intervention` flag in `df_temp` based on ab_groups min/max per encounter
    - Rename `value` columns to metric-specific names
    - Basic filtering:
        - temperature: remove 0 and values < 34
        - heart_rate: set 0 to NaN

    Parameters
    ----------
    strict:
        If True, raises exceptions on schema issues. If False, logs warnings and continues
        where possible.
    copy:
        If True, defensively copies inputs before mutating.

    Returns
    -------
    PreprocessedData
        A dataclass with named dataframe attributes.
    """
    # Defensive copies
    if copy:
        df_temp = df_temp.copy()
        df_heart_rates = df_heart_rates.copy()
        df_so2 = df_so2.copy()
        df_bp = df_bp.copy()
        df_crp = df_crp.copy()
        df_bili = df_bili.copy()
        df_leua = df_leua.copy()
        df_krea = df_krea.copy()
        df_hb = df_hb.copy()
        df_ab_groups = df_ab_groups.copy()

    def handle_error(msg: str, exc: Exception) -> None:
        if strict:
            raise
        LOGGER.warning("%s: %s", msg, exc, exc_info=True)

    # Drop NaNs for specific lab dfs (as in your code)
    try:
        df_bili = _dropna_in_value(df_bili)
        df_crp = _dropna_in_value(df_crp)
        df_hb = _dropna_in_value(df_hb)
        df_krea = _dropna_in_value(df_krea)
        df_leua = _dropna_in_value(df_leua)
    except Exception as e:
        handle_error("Error dropping NaNs for lab dataframes", e)

    # Add timestamps + recorded_date
    for name, df in [
        ("df_temp", df_temp),
        ("df_heart_rates", df_heart_rates),
        ("df_so2", df_so2),
        ("df_bp", df_bp),
        ("df_bili", df_bili),
        ("df_crp", df_crp),
        ("df_hb", df_hb),
        ("df_krea", df_krea),
        ("df_leua", df_leua),
    ]:
        try:
            # Assign back to ensure we keep copies returned by helper
            if name == "df_temp":
                df_temp = _add_recorded_date(df_temp)
            elif name == "df_heart_rates":
                df_heart_rates = _add_recorded_date(df_heart_rates)
            elif name == "df_so2":
                df_so2 = _add_recorded_date(df_so2)
            elif name == "df_bp":
                df_bp = _add_recorded_date(df_bp)
            elif name == "df_bili":
                df_bili = _add_recorded_date(df_bili)
            elif name == "df_crp":
                df_crp = _add_recorded_date(df_crp)
            elif name == "df_hb":
                df_hb = _add_recorded_date(df_hb)
            elif name == "df_krea":
                df_krea = _add_recorded_date(df_krea)
            elif name == "df_leua":
                df_leua = _add_recorded_date(df_leua)
        except Exception as e:
            handle_error(f"Error processing timestamps for {name}", e)

    # Antibiotic window -> intervention flag on df_temp
    try:
        _require_columns(df_ab_groups, ["encounter_id", "recorded_time"], name="df_ab_groups")
        df_ab_groups = _add_recorded_date(df_ab_groups, time_column="recorded_time")  # coerces time
        df_ab_grouped = (
            df_ab_groups.groupby("encounter_id")["recorded_time"]
            .agg(min_timestamp="min", max_timestamp="max")
            .reset_index()
        )

        _require_columns(df_temp, ["encounter_id", "recorded_time"], name="df_temp")
        df_temp = df_temp.merge(df_ab_grouped, on="encounter_id", how="left")

        in_window = (
            (df_temp["recorded_time"] >= df_temp["min_timestamp"]) &
            (df_temp["recorded_time"] <= df_temp["max_timestamp"])
        )
        df_temp["intervention"] = in_window.fillna(False).astype(int)
        df_temp = df_temp.drop(columns=["min_timestamp", "max_timestamp"])
    except Exception as e:
        handle_error("Error creating intervention flag from df_ab_groups", e)

    # Rename `value` columns
    try:
        df_heart_rates = _rename_value_column(df_heart_rates, "heart_rate")
        df_so2 = _rename_value_column(df_so2, "so2")
        df_bp = _rename_value_column(df_bp, "mean_arterial_pressure")
        df_bili = _rename_value_column(df_bili, "bili")
        df_crp = _rename_value_column(df_crp, "crp")
        df_hb = _rename_value_column(df_hb, "hb")
        df_krea = _rename_value_column(df_krea, "krea")
        df_leua = _rename_value_column(df_leua, "leua")
    except Exception as e:
        handle_error("Error renaming columns", e)

    # Filtering / cleanup rules
    try:
        _require_columns(df_temp, ["value"], name="df_temp")
        df_temp = df_temp[(df_temp["value"] != 0) & (df_temp["value"] >= 34)]
    except Exception as e:
        handle_error("Error filtering temperature values", e)

    try:
        _require_columns(df_heart_rates, ["heart_rate"], name="df_heart_rates")
        df_heart_rates.loc[df_heart_rates["heart_rate"] == 0, "heart_rate"] = np.nan
    except Exception as e:
        handle_error("Error cleaning heart rate values", e)

    return PreprocessedData(
        temp=df_temp,
        heart_rates=df_heart_rates,
        so2=df_so2,
        bp=df_bp,
        crp=df_crp,
        bili=df_bili,
        leua=df_leua,
        krea=df_krea,
        hb=df_hb,
        ab_groups=df_ab_groups,
    )


# In[ ]:


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _as_utc_datetime(s: pd.Series) -> pd.Series:
    """Convert a Series to UTC datetime, coercing invalid parses to NaT."""
    return pd.to_datetime(s, utc=True, errors="coerce")


def _compute_time_lag_flags(group: pd.DataFrame) -> pd.DataFrame:
    """
    For one encounter_id group, compute boolean time-lag flags relative to the
    earliest 'intervention == 1' timestamp in that group.

    If no intervention timestamp exists, all flags are False.
    """
    intervention_time = group.loc[group["intervention"] == 1, "recorded_time"].min()
    out = pd.DataFrame(index=group.index)

    if pd.isna(intervention_time):
        out["time_lag_2"] = False            # within 0-24h after intervention
        out["time_lag_1"] = False            # within 24-48h after intervention
        out["time_lag_target"] = False       # within 48-72h after intervention
        out["time_lag_prior_24h"] = False    # 24h before intervention
        return out

    dt = group["recorded_time"] - intervention_time
    seconds = dt.dt.total_seconds()

    out["time_lag_2"] = (seconds >= 0) & (seconds <= 24 * 60 * 60)
    out["time_lag_1"] = (seconds > 24 * 60 * 60) & (seconds <= 48 * 60 * 60)
    out["time_lag_target"] = (seconds > 48 * 60 * 60) & (seconds <= 72 * 60 * 60)

    prior_start = intervention_time - pd.Timedelta(hours=24)
    out["time_lag_prior_24h"] = (group["recorded_time"] >= prior_start) & (
        group["recorded_time"] < intervention_time
    )
    return out


def _merge_asof_by_encounter(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    value_col: str,
    time_col: str = "recorded_time",
    by_col: str = "encounter_id",
    direction: str = "backward",
    tolerance: pd.Timedelta | None = None,
    allow_exact_matches: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    As-of merge within encounter_id.

    - Drops rows with NaT in the merge key, because merge_asof forbids null keys.
    - Sorts by [time_col, by_col] to satisfy pandas' global monotonic requirement.
    """
    _require_columns(base, [by_col, time_col], name="base")
    _require_columns(other, [by_col, time_col, value_col], name=f"other({value_col})")

    base2 = base.copy()
    other2 = other[[by_col, time_col, value_col]].copy()

    # Drop NaT keys (cannot be aligned)
    base2 = base2.loc[base2[time_col].notna()].copy()
    other2 = other2.loc[other2[time_col].notna()].copy()

    if other2.empty:
        msg = f"All rows in other({value_col}) have NaT in '{time_col}' after parsing; merge skipped."
        if strict:
            raise ValueError(msg)
        LOGGER.warning(msg)
        return base.copy()

    # Sort by time first (required by merge_asof)
    base_sorted = base2.sort_values([time_col, by_col])
    other_sorted = other2.sort_values([time_col, by_col])

    merged = pd.merge_asof(
        base_sorted,
        other_sorted,
        on=time_col,
        by=by_col,
        direction=direction,
        tolerance=tolerance,
        allow_exact_matches=allow_exact_matches,
    )

    return merged.sort_values([by_col, time_col])




from __future__ import annotations

from typing import List, Sequence, Tuple
import logging
import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def create_time_lags_and_merge_features(
    df_temp: pd.DataFrame,
    df_heart_rates: pd.DataFrame,
    df_so2: pd.DataFrame,
    df_bp: pd.DataFrame,
    df_bili: pd.DataFrame,
    df_crp: pd.DataFrame,
    df_hb: pd.DataFrame,
    df_krea: pd.DataFrame,
    df_leua: pd.DataFrame,
    df_ab_groups: pd.DataFrame,  # kept for API compatibility; not used here
    *,
    strict: bool = True,
    copy: bool = True,
    forward_fill_cols: Sequence[str] = ("crp", "bili", "krea", "leua", "hb", "heart_rate"),
) -> pd.DataFrame:
    """
    Create time-lag flags relative to the earliest intervention==1 timestamp per encounter,
    then as-of merge additional measurement streams onto the df_temp timeline.

    Notes
    -----
    - merge_asof requires global sorting by recorded_time, which is handled in _merge_asof_by_encounter.
    - This function assumes df_temp contains columns:
        encounter_id, subject_reference, recorded_time, value, intervention
      and that the other dataframes contain:
        encounter_id, recorded_time, and their metric column (e.g., crp, bili, ...)
    """
    def handle_error(msg: str, exc: Exception) -> None:
        if strict:
            raise
        LOGGER.warning("%s: %s", msg, exc, exc_info=True)

    if copy:
        df_temp = df_temp.copy()
        df_heart_rates = df_heart_rates.copy()
        df_so2 = df_so2.copy()
        df_bp = df_bp.copy()
        df_bili = df_bili.copy()
        df_crp = df_crp.copy()
        df_hb = df_hb.copy()
        df_krea = df_krea.copy()
        df_leua = df_leua.copy()

    # --- Validate and normalize timestamps on df_temp ---
    try:
        _require_columns(
            df_temp,
            ["encounter_id", "subject_reference", "recorded_time", "value", "intervention"],
            name="df_temp",
        )
        df_temp["recorded_time"] = pd.to_datetime(df_temp["recorded_time"], utc=True, errors="coerce")
    except Exception as e:
        handle_error("df_temp schema/timestamp parsing failed", e)

    # --- Normalize timestamps on covariate dfs (coerce invalid to NaT) ---
    for name, df in [
        ("df_heart_rates", df_heart_rates),
        ("df_so2", df_so2),
        ("df_bp", df_bp),
        ("df_bili", df_bili),
        ("df_crp", df_crp),
        ("df_hb", df_hb),
        ("df_krea", df_krea),
        ("df_leua", df_leua),
    ]:
        try:
            _require_columns(df, ["encounter_id", "recorded_time"], name=name)
            df["recorded_time"] = pd.to_datetime(df["recorded_time"], utc=True, errors="coerce")
        except Exception as e:
            handle_error(f"{name} schema/timestamp parsing failed", e)

    # --- Compute time-lag flags per encounter ---
    try:
        lag_flags = (
            df_temp.groupby("encounter_id", group_keys=False)
            .apply(_compute_time_lag_flags)
            .sort_index()
        )
        df_temp = df_temp.join(lag_flags)
    except Exception as e:
        handle_error("Computing time-lag flags failed", e)

    # --- Base dataframe aligned to df_temp timeline ---
    base_cols = [
        "encounter_id",
        "subject_reference",
        "recorded_time",
        "value",
        "intervention",
        "time_lag_1",
        "time_lag_2",
        "time_lag_target",
    ]
    if "time_lag_prior_24h" in df_temp.columns:
        base_cols.append("time_lag_prior_24h")

    base_df = df_temp[base_cols].copy()

    # --- Merge covariates via as-of merges (FIXED: only inside the loop) ---
    merges: List[Tuple[pd.DataFrame, str]] = [
        (df_crp, "crp"),
        (df_heart_rates, "heart_rate"),
        (df_bp, "mean_arterial_pressure"),
        (df_so2, "so2"),
        (df_bili, "bili"),
        (df_hb, "hb"),
        (df_krea, "krea"),
        (df_leua, "leua"),
    ]

    for other_df, value_col in merges:
        try:
            if value_col not in other_df.columns:
                raise ValueError(
                    f"Expected column '{value_col}' in dataframe for merge, "
                    f"but got columns={other_df.columns.tolist()}"
                )
            base_df = _merge_asof_by_encounter(
                base_df,
                other_df,
                value_col=value_col,
                strict=strict,
            )
        except Exception as e:
            handle_error(f"As-of merge failed for '{value_col}'", e)

    # --- Forward fill within encounter after merge (optional; mirrors old behavior) ---
    try:
        for col in forward_fill_cols:
            if col in base_df.columns:
                base_df[col] = base_df.groupby("encounter_id")[col].ffill()
    except Exception as e:
        handle_error("Forward fill step failed", e)

    return base_df



# In[ ]:


FOURIER_FEATURE_COLUMNS = [
    "fourier_magnitude",
    "fourier_phase",
    "fourier_real_part",
    "fourier_imaginary_part",
    "fourier_mean_magnitude",
    "fourier_std_dev_magnitude",
    "fourier_max_magnitude",
    "fourier_min_magnitude",
]


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def create_fourier_features_for_group(
    group: pd.DataFrame,
    *,
    value_col: str = "value",
    lag_cols: tuple[str, ...] = ("time_lag_1", "time_lag_2"),
    strict: bool = True,
) -> pd.Series:
    """
    Compute Fourier-based features for a single grouped time series.

    The Fourier transform is computed on `value_col` restricted to rows where any of
    `lag_cols` is True (default: time_lag_1 or time_lag_2).

    Returns a Series with:
    - vector-valued outputs (arrays): magnitude, phase, real_part, imaginary_part
    - scalar summary stats: mean/std/max/min of magnitudes
    
    Parameters
    ----------
    strict:
        If True, raise exceptions. If False, log and return NaNs.
    """
    def fail(msg: str, exc: Optional[Exception] = None) -> pd.Series:
        if exc is not None:
            if strict:
                raise exc
            LOGGER.warning("%s: %s", msg, exc, exc_info=True)
        else:
            LOGGER.info("%s", msg)

        return pd.Series([np.nan] * len(FOURIER_FEATURE_COLUMNS), index=FOURIER_FEATURE_COLUMNS)

    try:
        _require_columns(group, [value_col, *lag_cols], name="group")
    except Exception as e:
        return fail("Missing required columns for Fourier features", e)

    # Ensure boolean lags and numeric values
    mask = False
    for c in lag_cols:
        mask = mask | group[c].astype(bool)

    values = pd.to_numeric(group.loc[mask, value_col], errors="coerce").dropna()

    if values.empty:
        return fail("No valid values in selected lag window")

    try:
        # FFT expects an array-like numeric sequence
        fft = np.fft.fft(values.to_numpy())

        magnitude = np.abs(fft)
        phase = np.angle(fft)
        real_part = np.real(fft)
        imaginary_part = np.imag(fft)

        # Scalar summaries
        mean_mag = float(np.mean(magnitude))
        std_mag = float(np.std(magnitude))
        max_mag = float(np.max(magnitude))
        min_mag = float(np.min(magnitude))

        return pd.Series(
            [
                magnitude,
                phase,
                real_part,
                imaginary_part,
                mean_mag,
                std_mag,
                max_mag,
                min_mag,
            ],
            index=FOURIER_FEATURE_COLUMNS,
        )

    except Exception as e:
        return fail("Error computing Fourier transform/features", e)


# In[ ]:


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def ffill_impute_within_encounter(
    df: pd.DataFrame,
    *,
    group_col: str = "encounter_id",
    time_col: Optional[str] = "recorded_time",
    columns: Optional[Sequence[str]] = None,
    exclude: Sequence[str] = ("value",),
    limit: Optional[int] = None,
    copy: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Forward-fill imputation within each encounter (no mean/median/global statistics).

    This avoids train/test leakage from global statistics. It can still leak *future*
    info if your rows are not time-ordered, so sort by (group_col, time_col) when
    time_col is provided and present.

    Parameters
    ----------
    df:
        Input dataframe.
    group_col:
        Column defining groups (default: encounter_id).
    time_col:
        Optional time column used for sorting before forward fill (default: recorded_time).
        If None, no sorting is performed.
    columns:
        Columns to forward-fill. If None, all numeric columns except those in `exclude`.
    exclude:
        Columns to exclude when `columns` is None (default excludes 'value').
    limit:
        Maximum number of consecutive NaNs to fill (pandas ffill limit). If None, unlimited.
    copy:
        If True, return a copy; if False, mutate and return df.
    strict:
        If True, raise on missing required columns; if False, log and best-effort proceed.

    Returns
    -------
    pd.DataFrame
        Dataframe with forward-filled values within each encounter.
    """
    try:
        _require_columns(df, [group_col], name="df")
    except Exception as e:
        if strict:
            raise
        LOGGER.warning("Missing group column '%s': %s", group_col, e, exc_info=True)
        return df.copy() if copy else df

    out = df.copy() if copy else df

    # Determine which columns to ffill
    if columns is None:
        numeric_cols = out.select_dtypes(include=["number"]).columns.tolist()
        cols_to_fill = [c for c in numeric_cols if c not in set(exclude)]
    else:
        cols_to_fill = list(columns)

    # Keep only columns that exist
    missing_cols = [c for c in cols_to_fill if c not in out.columns]
    if missing_cols:
        msg = f"Skipping missing columns for ffill: {missing_cols}"
        if strict:
            raise ValueError(msg)
        LOGGER.warning(msg)
    cols_to_fill = [c for c in cols_to_fill if c in out.columns]

    if not cols_to_fill:
        LOGGER.info("No columns selected for forward-fill imputation.")
        return out

    # Sort for time-correct forward fill
    if time_col is not None and time_col in out.columns:
        out = out.sort_values([group_col, time_col])

    # Forward fill within each encounter
    out[cols_to_fill] = out.groupby(group_col)[cols_to_fill].ffill(limit=limit)

    return out


# In[ ]:


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def add_statistical_lag_features(
    df: pd.DataFrame,
    *,
    group_col: str = "encounter_id",
    lag1_col: str = "time_lag_1",
    lag2_col: str = "time_lag_2",
    target_lag_col: str = "time_lag_target",
    max_columns: Sequence[str] = (
        "value",
        "krea",
        "bili",
        "crp",
        "leua",
        "hb",
        "heart_rate",
        "so2",
        "mean_arterial_pressure",
    ),
    value_col: str = "value",
    copy: bool = True,
    strict: bool = True,
    drop_missing_required: bool = True,
) -> pd.DataFrame:
    """
    Add per-encounter statistical features computed over time-lag windows.

    This is a reviewer-friendly rewrite of your `lag_features` function.

    Features added (per encounter_id):
    - For each column in `max_columns`:
        - {lag1_col}_{col}_max
        - {lag2_col}_{col}_max
    - For `value_col` only:
        - {lag1_col}_bt_mean   (mean of value in lag1 window)
        - {lag2_col}_bt_mean   (mean of value in lag2 window) 
        - {target_lag_col}_bt_max (max of value in target window)

    Notes
    -----
    - The created features are constant within each encounter and are merged back into df.
    - Avoids repeated merges in a loop by building aggregated tables once per window.

    Parameters
    ----------
    drop_missing_required:
        If True, drop rows that are missing required lag-window features. If False, keep rows (features may be NaN).

    Returns
    -------
    pd.DataFrame
        Original dataframe with additional lag-feature columns.
    """
    def handle_error(msg: str, exc: Exception) -> pd.DataFrame:
        if strict:
            raise
        LOGGER.warning("%s: %s", msg, exc, exc_info=True)
        return df.copy() if copy else df

    try:
        _require_columns(df, [group_col, lag1_col, lag2_col, target_lag_col], name="df")
        _require_columns(df, [value_col], name="df")
        missing_max_cols = [c for c in max_columns if c not in df.columns]
        if missing_max_cols:
            raise ValueError(f"df is missing columns listed in max_columns: {missing_max_cols}")
    except Exception as e:
        return handle_error("Schema validation failed for lag feature computation", e)

    out = df.copy() if copy else df

    # Ensure boolean lags
    out[lag1_col] = out[lag1_col].astype(bool)
    out[lag2_col] = out[lag2_col].astype(bool)
    out[target_lag_col] = out[target_lag_col].astype(bool)

    # ---- MAX features for lag1 and lag2 windows ----
    try:
        lag1_max = (
            out.loc[out[lag1_col], [group_col, *max_columns]]
            .groupby(group_col, as_index=False)
            .max()
            .rename(columns={c: f"{lag1_col}_{c}_max" for c in max_columns})
        )

        lag2_max = (
            out.loc[out[lag2_col], [group_col, *max_columns]]
            .groupby(group_col, as_index=False)
            .max()
            .rename(columns={c: f"{lag2_col}_{c}_max" for c in max_columns})
        )

        out = out.merge(lag1_max, on=group_col, how="left").merge(lag2_max, on=group_col, how="left")
    except Exception as e:
        return handle_error("Failed to compute/merge lag1/lag2 max features", e)

    # ---- Mean features for VALUE only ----
    try:
        lag1_value_mean = (
            out.loc[out[lag1_col], [group_col, value_col]]
            .groupby(group_col, as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: f"{lag1_col}_bt_mean"})
        )

        lag2_value_mean = (
            out.loc[out[lag2_col], [group_col, value_col]]
            .groupby(group_col, as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: f"{lag2_col}_bt_mean"})
        )

        out = out.merge(lag1_value_mean, on=group_col, how="left").merge(lag2_value_mean, on=group_col, how="left")
    except Exception as e:
        return handle_error("Failed to compute/merge lag1 mean and lag2 mean features", e)

    # ---- Target window max for VALUE ----
    try:
        target_value_max = (
            out.loc[out[target_lag_col], [group_col, value_col]]
            .groupby(group_col, as_index=False)[value_col]
            .max()
            .rename(columns={value_col: f"{target_lag_col}_bt_max"})
        )
        out = out.merge(target_value_max, on=group_col, how="left")
    except Exception as e:
        return handle_error("Failed to compute/merge target max feature", e)

    # ---- Drop rows missing key features ----
    if drop_missing_required:
        required = [f"{lag1_col}_{value_col}_max", f"{lag2_col}_{value_col}_max"]
        pass

    # ---- Final renames to keep backward compatible names ----
    rename_map: Mapping[str, str] = {
        f"{lag1_col}_{value_col}_max": f"{lag1_col}_bt_max",
        f"{lag2_col}_{value_col}_max": f"{lag2_col}_bt_max",
        f"{lag1_col}_bt_mean": f"{lag1_col}_bt_mean",
        f"{lag2_col}_bt_mean": f"{lag2_col}_bt_mean",
        f"{target_lag_col}_bt_max": f"{target_lag_col}_bt_max",
    }
    out = out.rename(columns=rename_map)

    if drop_missing_required:
        out = out.dropna(subset=[f"{lag1_col}_bt_max", f"{lag2_col}_bt_max"], how="any")

    return out


# In[ ]:


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _time_of_day_bucket(hours: pd.Series) -> pd.Series:
    """
    Bucket hours into {morning, afternoon, evening, night}.
    """
    # Vectorized (faster than apply with lambda)
    return pd.cut(
        hours,
        bins=[-1, 4, 10, 16, 22, 24],
        labels=["night", "morning", "afternoon", "evening", "night"],
        right=True,
        include_lowest=True,
        ordered=False,
    )


def _compute_fever_stats_for_lag(
    df: pd.DataFrame,
    *,
    lag_col: str,
    group_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    fever_threshold: float = 38.0,
    valid_bt_threshold: float = 35.0,
) -> pd.DataFrame:
    """
    Compute per-encounter fever/temperature summary stats restricted to rows where df[lag_col] is True.

    Returns a dataframe indexed by encounter_id with columns:
    - fever_percent
    - fever_points
    - fever_change_pct
    - last_bt
    - fever_diff
    - fever_evening / fever_afternoon / fever_morning / fever_night (max if any value >= valid_bt_threshold else NaN)
    - fever_std
    - fever_range
    """
    df_lag = df.loc[df[lag_col]].copy()
    if df_lag.empty:
        # Return an empty frame with the expected columns
        cols = [
            "fever_percent",
            "fever_points",
            "fever_change_pct",
            "last_bt",
            "fever_diff",
            "fever_evening",
            "fever_afternoon",
            "fever_morning",
            "fever_night",
            "fever_std",
            "fever_range",
        ]
        return pd.DataFrame(columns=cols).set_index(pd.Index([], name=group_col))

    # Ensure sorted for first/last calculations
    df_lag = df_lag.sort_values([group_col, time_col])

    # Core aggregates
    g = df_lag.groupby(group_col)[value_col]

    fever_percent = g.apply(lambda x: float((x >= fever_threshold).mean() * 100.0))
    fever_points = g.apply(lambda x: int((x >= fever_threshold).sum()))
    first_val = g.first()
    last_val = g.last()

    # Avoid division by zero
    fever_change_pct = (last_val - first_val) / first_val.replace(0, np.nan) * 100.0
    fever_diff = last_val - first_val

    fever_std = g.std()
    fever_range = g.max() - g.min()

    # Time-of-day maxima within lag window
    hours = df_lag[time_col].dt.hour
    tod = _time_of_day_bucket(hours)
    df_lag = df_lag.assign(time_of_day=tod)

    def _tod_max(name: str) -> pd.Series:
        sub = df_lag.loc[df_lag["time_of_day"] == name].groupby(group_col)[value_col]
        # max only if any >= valid_bt_threshold else NaN (matches your "if (x >= 35).any() else None")
        return sub.apply(lambda x: x.max() if (x >= valid_bt_threshold).any() else np.nan)

    fever_morning = _tod_max("morning")
    fever_afternoon = _tod_max("afternoon")
    fever_evening = _tod_max("evening")
    fever_night = _tod_max("night")

    out = pd.DataFrame(
        {
            "fever_percent": fever_percent,
            "fever_points": fever_points,
            "fever_change_pct": fever_change_pct,
            "last_bt": last_val,
            "fever_diff": fever_diff,
            "fever_evening": fever_evening,
            "fever_afternoon": fever_afternoon,
            "fever_morning": fever_morning,
            "fever_night": fever_night,
            "fever_std": fever_std,
            "fever_range": fever_range,
        }
    )
    out.index.name = group_col
    return out


def _compute_cross_lag_features(
    df: pd.DataFrame,
    *,
    group_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    lag1_col: str = "time_lag_1",
    lag2_col: str = "time_lag_2",
) -> pd.DataFrame:
    """
    Features computed over combined lag1/lag2 data:
    - skewness over union(lag1, lag2)
    - fever_change_lag_all: ((first value in lag1) - (last value in lag2)) / first_lag1 * 100

    If lag1 or lag2 is missing for an encounter, fever_change_lag_all is NaN.
    """
    df_u = df.loc[df[lag1_col] | df[lag2_col]].sort_values([group_col, time_col])

    if df_u.empty:
        return pd.DataFrame(columns=["fever_change_lag_all", "skewness_lag_all"]).set_index(
            pd.Index([], name=group_col)
        )

    skewness = df_u.groupby(group_col)[value_col].skew().rename("skewness_lag_all")

    def _change(group: pd.DataFrame) -> float:
        g1 = group.loc[group[lag1_col], value_col]
        g2 = group.loc[group[lag2_col], value_col]
        if g1.empty or g2.empty:
            return np.nan
        denom = g1.iloc[0]
        if denom == 0 or pd.isna(denom):
            return np.nan
        return float(((g1.iloc[0] - g2.iloc[-1]) / denom) * 100.0)

    change = df_u.groupby(group_col).apply(_change).rename("fever_change_lag_all")

    out = pd.concat([change, skewness], axis=1)
    out.index.name = group_col
    return out


def _compute_trend_per_encounter(
    df: pd.DataFrame,
    *,
    group_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    use_mask: Optional[pd.Series] = None,
) -> pd.Series:
    """
    Compute a simple linear trend (slope) of value over time per encounter.

    Implementation uses numpy polyfit on seconds since first observation in group.
    """
    data = df.copy()
    if use_mask is not None:
        data = data.loc[use_mask]

    if data.empty:
        return pd.Series(dtype="float64", name="trend")

    data = data.sort_values([group_col, time_col])

    def _trend(group: pd.DataFrame) -> float:
        y = pd.to_numeric(group[value_col], errors="coerce").to_numpy()
        t = group[time_col].view("int64") / 1e9  # seconds since epoch (float)
        # normalize to improve conditioning
        t = t - t.min()
        mask = np.isfinite(y) & np.isfinite(t)
        y = y[mask]
        t = t[mask]
        if len(y) < 2:
            return np.nan
        # slope of y ~ a*t + b
        a, _b = np.polyfit(t, y, deg=1)
        return float(a)

    return data.groupby(group_col, group_keys=False).apply(_trend).rename("trend")


def create_features(
    df: pd.DataFrame,
    *,
    group_col: str = "encounter_id",
    time_col: str = "recorded_time",
    value_col: str = "value",
    lag1_col: str = "time_lag_1",
    lag2_col: str = "time_lag_2",
    target_lag_col: str = "time_lag_target",
    fever_threshold: float = 38.0,
    min_required_bt_max_col: str = "time_lag_1_bt_max",
    copy: bool = True,
    strict: bool = True,
    fillna_value: float = 0.0,
) -> pd.DataFrame:
    """
    Create additional fever/temperature-derived statistical features.

    Returns the input dataframe with additional feature columns.
    """
    def handle_error(msg: str, exc: Exception) -> pd.DataFrame:
        if strict:
            raise
        LOGGER.warning("%s: %s", msg, exc, exc_info=True)
        return df.copy() if copy else df

    try:
        out = df.copy() if copy else df

        _require_columns(
            out,
            [group_col, time_col, value_col, lag1_col, lag2_col, target_lag_col],
            name="df",
        )
        if min_required_bt_max_col not in out.columns:
            raise ValueError(f"Missing required column '{min_required_bt_max_col}' for filtering.")
        out[time_col] = pd.to_datetime(out[time_col], utc=True, errors="coerce")
    except Exception as e:
        return handle_error("Schema validation / datetime parsing failed", e)

    # Ensure boolean lag cols
    out[lag1_col] = out[lag1_col].astype(bool)
    out[lag2_col] = out[lag2_col].astype(bool)
    out[target_lag_col] = out[target_lag_col].astype(bool)

    # ---- Per-lag fever stats ----
    try:
        lag1_stats = _compute_fever_stats_for_lag(
            out,
            lag_col=lag1_col,
            group_col=group_col,
            time_col=time_col,
            value_col=value_col,
            fever_threshold=fever_threshold,
        ).add_prefix("lag1_")

        lag2_stats = _compute_fever_stats_for_lag(
            out,
            lag_col=lag2_col,
            group_col=group_col,
            time_col=time_col,
            value_col=value_col,
            fever_threshold=fever_threshold,
        ).add_prefix("lag2_")

        cross_stats = _compute_cross_lag_features(
            out,
            group_col=group_col,
            time_col=time_col,
            value_col=value_col,
            lag1_col=lag1_col,
            lag2_col=lag2_col,
        )

        # Merge once per table
        out = out.merge(lag1_stats, left_on=group_col, right_index=True, how="left")
        out = out.merge(lag2_stats, left_on=group_col, right_index=True, how="left")
        out = out.merge(cross_stats, left_on=group_col, right_index=True, how="left")

        # Backward-compatible column names
        rename = {
            "lag1_fever_points": "fever_points_lag_1",
            "lag1_fever_percent": "fever_percent_lag_1",
            "lag1_fever_change_pct": "fever_change_lag_1",
            "lag1_last_bt": "last_bt_fever_lag_1",
            "lag1_fever_diff": "fever_diff_lag_1",
            "lag1_fever_evening": "fever_lag_1_evening",
            "lag1_fever_afternoon": "fever_lag_1_afternoon",
            "lag1_fever_morning": "fever_lag_1_morning",
            "lag1_fever_night": "fever_lag_1_night",
            "lag1_fever_std": "fever_variability_lag_1",
            "lag1_fever_range": "fever_range_lag_1",
            "lag2_fever_points": "fever_points_lag_2",
            "lag2_fever_percent": "fever_percent_lag_2",
            "lag2_fever_change_pct": "fever_change_lag_2",
            "lag2_last_bt": "last_bt_fever_lag_2",
            "lag2_fever_diff": "fever_diff_lag_2",
            "lag2_fever_evening": "fever_lag_2_evening",
            "lag2_fever_afternoon": "fever_lag_2_afternoon",
            "lag2_fever_morning": "fever_lag_2_morning",
            "lag2_fever_night": "fever_lag_2_night",
            "lag2_fever_std": "fever_variability_lag_2",
            "lag2_fever_range": "fever_range_lag_2",
        }
        out = out.rename(columns=rename)

        # Fill NaNs for these engineered features (matches your fillna(0) behavior)
        engineered_cols = list(rename.values()) + ["fever_change_lag_all", "skewness_lag_all"]
        for c in engineered_cols:
            if c in out.columns:
                out[c] = out[c].fillna(fillna_value)

    except Exception as e:
        return handle_error("Failed computing/merging fever statistics", e)

    # ---- Filters that define your analytic cohort ----
    try:
        out = out.loc[out[min_required_bt_max_col] >= fever_threshold]
    except Exception as e:
        return handle_error(f"Filtering on {min_required_bt_max_col} failed", e)

    # Keep only encounters that have at least one target + one lag1 observation
    try:
        # Efficient group-level filters using transform
        has_target = out.groupby(group_col)[target_lag_col].transform("sum") >= 1
        has_lag1 = out.groupby(group_col)[lag1_col].transform("sum") >= 1
        out = out.loc[has_target & has_lag1]
    except Exception as e:
        return handle_error("Filtering encounters by required lag counts failed", e)

    # ---- Time-of-day label ----
    try:
        hours = out[time_col].dt.hour
        out["time_of_day"] = pd.cut(
            hours,
            bins=[-1, 4, 10, 16, 23],
            labels=["night", "morning", "afternoon", "evening"],
            include_lowest=True,
        ).astype(str)
    except Exception as e:
        return handle_error("Creating time_of_day feature failed", e)

    # ---- Trend feature over union(lag1, lag2) ----
    try:
        mask = out[lag1_col] | out[lag2_col]
        trend = _compute_trend_per_encounter(
            out,
            group_col=group_col,
            time_col=time_col,
            value_col=value_col,
            use_mask=mask,
        )
        out = out.merge(trend, left_on=group_col, right_index=True, how="left")
    except Exception as e:
        return handle_error("Computing trend feature failed", e)

    return out


# In[ ]:


def _require_columns(df: pd.DataFrame, cols: Iterable[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} is missing required columns: {missing}")


def _normalize_icd10(code: str) -> str:
    """
    Normalize ICD-10(-CM) codes for matching:
    - uppercase
    - remove dots and non-alphanumeric chars
    HCUP expects ICD-10-CM diagnosis codes without decimals. :contentReference[oaicite:2]{index=2}
    """
    if code is None:
        return ""
    code = str(code).upper()
    return re.sub(r"[^0-9A-Z]", "", code)


@dataclass(frozen=True)
class ElixhauserResult:
    """Outputs of Elixhauser assignment."""
    indicators: pd.DataFrame  # one row per subject_reference, one column per comorbidity (0/1)
    score: pd.DataFrame       # subject_reference + elixhauser_score (sum of indicators)


def _build_elixhauser_mapping_from_hcup_reference(reference_xlsx_path: str) -> pd.DataFrame:
    """
    Robustly load HCUP Elixhauser ICD-10-CM reference Excel and return a long mapping table:
        comorbidity | match_type | match_value

    Handles:
    - duplicated column names
    - multiple ICD/code columns per row (stacks them)
    - varying sheet/column naming conventions
    """
    xls = pd.ExcelFile(reference_xlsx_path)

    best_sheet = None
    best_df = None
    best_score = -1

    # choose best-looking sheet by heuristic signals
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        cols = [str(c).lower() for c in df.columns]
        score = sum(any(k in c for k in ("comorb", "elix", "measure", "condition")) for c in cols) + \
                sum(any(k in c for k in ("icd", "dx", "diagn", "code")) for c in cols)
        if score > best_score:
            best_score = score
            best_sheet = sheet
            best_df = df

    if best_df is None:
        raise ValueError("Could not read any sheets from the HCUP reference Excel.")

    ref = best_df.copy()

    # helper: find comorbidity col (single best)
    def find_first_col(keys: Tuple[str, ...]) -> Optional[str]:
        for c in ref.columns:
            cl = str(c).lower()
            if any(k in cl for k in keys):
                return c
        return None

    comorb_col = find_first_col(("comorb", "elix", "measure", "condition", "category"))
    if comorb_col is None:
        raise ValueError(
            f"Could not infer comorbidity column from sheet '{best_sheet}'. "
            f"Columns seen: {ref.columns.tolist()}"
        )

    # helper: find ALL code cols (there may be multiple)
    code_cols = [
        c for c in ref.columns
        if any(k in str(c).lower() for k in ("icd", "dx", "diagn", "code"))
        and c != comorb_col
    ]
    if not code_cols:
        raise ValueError(
            f"Could not infer any ICD/code columns from sheet '{best_sheet}'. "
            f"Columns seen: {ref.columns.tolist()}"
        )

    # Keep only relevant columns, then stack all code columns into one "code" column
    sub = ref[[comorb_col] + code_cols].copy()
    sub = sub.rename(columns={comorb_col: "comorbidity"})

    # Melt to long format: one code per row
    long = sub.melt(
        id_vars=["comorbidity"],
        value_vars=code_cols,
        var_name="code_source",
        value_name="code",
    )

    long = long.dropna(subset=["comorbidity", "code"]).copy()
    long["comorbidity"] = long["comorbidity"].astype(str).str.strip()
    long["code"] = long["code"].astype(str).str.strip()
    long = long.loc[(long["comorbidity"] != "") & (long["code"] != "")].copy()

    # Normalize match rules
    long["code_raw"] = long["code"].astype(str)

    def to_match_rule(s: str) -> Tuple[str, str]:
        s = s.strip().upper()
        # treat as regex if it contains regex-ish metacharacters
        if any(ch in s for ch in ["^", "$", "[", "]", "(", ")", "|", "+", "?", "\\"]):
            return ("regex", s)
        # otherwise prefix match on normalized ICD-10-CM (no dots)
        from re import sub as re_sub
        normalized = re_sub(r"[^0-9A-Z]", "", s)
        return ("prefix", normalized)

    rules = long["code_raw"].apply(to_match_rule)
    long["match_type"] = rules.apply(lambda x: x[0])
    long["match_value"] = rules.apply(lambda x: x[1])

    long = long.loc[long["match_value"].astype(str).str.len() > 0].copy()

    # Deduplicate
    mapping = long[["comorbidity", "match_type", "match_value"]].drop_duplicates()

    return mapping

def _normalize_icd10(code: object) -> str:
    """
    Normalize ICD-10(-CM) diagnosis codes for matching:
    - uppercase
    - remove dots and any non-alphanumeric characters
    Example: 'I50.9' -> 'I509'
    """
    if code is None or (isinstance(code, float) and np.isnan(code)):
        return ""
    s = str(code).upper().strip()
    return re.sub(r"[^0-9A-Z]", "", s)


def calculate_elixhauser_score(
    df: pd.DataFrame,
    df_conds: pd.DataFrame,
    *,
    reference_xlsx_path: str,
    subject_col: str = "subject_reference",
    dx_col: str = "conditions",
    ref_code_col: str = "ICD-10-CM Diagnosis",
    return_indicators: bool = True,
    strict: bool = True,
) -> pd.DataFrame:
    """
    Calculate Elixhauser comorbidities using a HCUP reference table that directly maps
    ICD-10-CM diagnosis codes -> comorbidity indicators (0/1).

    This version is designed for the exact structure of your uploaded file:
    /mnt/data/HCUP_ELIXHAUSER_REFERENCE.xlsx

    Output:
    - merges 'elixhauser_score' onto df by subject_reference
    - optionally merges per-comorbidity indicator columns (0/1)

    Notes:
    - This computes an unweighted score: number of comorbidity categories present.
    - Codes not found in the reference are ignored (contribute 0).
    """
    def handle_error(msg: str, exc: Exception) -> pd.DataFrame:
        if strict:
            raise
        LOGGER.warning("%s: %s", msg, exc, exc_info=True)
        return df

    # Validate inputs
    try:
        _require_columns(df, [subject_col], name="df")
        _require_columns(df_conds, [subject_col, dx_col], name="df_conds")
    except Exception as e:
        return handle_error("Schema validation failed", e)

    # Load reference
    try:
        ref = pd.read_excel(reference_xlsx_path)
        _require_columns(ref, [ref_code_col], name="reference")
    except Exception as e:
        return handle_error("Failed to load HCUP Elixhauser reference Excel", e)

    # Identify comorbidity indicator columns:
    # everything except the code/description/meta columns
    meta_like = {ref_code_col, "ICD-10-CM Code Description", "# Comorbidities"}
    comorb_cols = [c for c in ref.columns if c not in meta_like]

    if not comorb_cols:
        return handle_error(
            "No comorbidity indicator columns found in reference file. "
            f"Columns seen: {ref.columns.tolist()}",
            ValueError("Reference appears to have no comorbidity columns"),
        )

    # Normalize reference ICD codes
    ref_small = ref[[ref_code_col] + comorb_cols].copy()
    ref_small["icd_norm"] = ref_small[ref_code_col].map(_normalize_icd10)
    ref_small = ref_small.loc[ref_small["icd_norm"] != ""].drop_duplicates(subset=["icd_norm"])

    # Normalize patient conditions
    conds = df_conds[[subject_col, dx_col]].dropna().copy()
    conds["icd_norm"] = conds[dx_col].map(_normalize_icd10)
    conds = conds.loc[conds["icd_norm"] != ""].copy()

    # Join conditions -> comorbidity flags
    joined = conds.merge(ref_small.drop(columns=[ref_code_col]), on="icd_norm", how="left")

    # Replace NaN flags with 0 (unknown code => no comorbidity flags contributed)
    joined[comorb_cols] = joined[comorb_cols].fillna(0)

    # Ensure numeric 0/1
    for c in comorb_cols:
        joined[c] = pd.to_numeric(joined[c], errors="coerce").fillna(0).astype(int)

    # Aggregate to subject: any occurrence => present
    indicators = joined.groupby(subject_col, as_index=True)[comorb_cols].max()

    # Unweighted score = number of comorbidities present
    score = indicators.sum(axis=1).rename("elixhauser_score").astype(int).to_frame()

    # Merge onto df
    out = df.copy()
    out = out.merge(score, left_on=subject_col, right_index=True, how="left")
    out["elixhauser_score"] = out["elixhauser_score"].fillna(0).astype(int)

    if return_indicators:
        out = out.merge(indicators, left_on=subject_col, right_index=True, how="left")
        out[comorb_cols] = out[comorb_cols].fillna(0).astype(int)

    return out



# In[ ]:


@dataclass(frozen=True)
class SplitResult:
    """Container for train/test split outputs."""
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_full: pd.DataFrame
    y_full: pd.Series
    scaler: Optional[MinMaxScaler]


def do_train_test_split(
    data: pd.DataFrame,
    *,
    feature_space: Sequence[str],
    target_col: str = "fever",
    test_size: float = 0.20,
    random_state: int = 23,
    stratify: bool = True,
    scale: Union[bool, str] = False,
    copy: bool = True,
    strict: bool = True,
) -> SplitResult:
    """
    Split a dataframe into train/test sets and optionally scale features.

    Reviewer-ready improvements over the original:
    - schema validation with clear errors
    - `scale` accepts bool or "yes"/"no"
    - returns a dataclass instead of a long tuple
    - avoids try/except prints; controlled by `strict` + logging

    IMPORTANT (leakage):
    - Scaling is fit on X_train only, then applied to X_test. This avoids leakage.

    Parameters
    ----------
    stratify:
        If True, stratify by y (recommended for class imbalance).
    scale:
        If True or "yes", apply MinMax scaling.
    """
    def handle_error(msg: str, exc: Exception) -> SplitResult:
        if strict:
            raise
        LOGGER.warning("%s: %s", msg, exc, exc_info=True)
        # Best-effort empty return
        empty_X = pd.DataFrame(index=data.index)
        empty_y = pd.Series(index=data.index, dtype="float64")
        return SplitResult(empty_X, empty_X, empty_y, empty_y, empty_X, empty_y, None)

    try:
        df = data.copy() if copy else data

        _require_columns(df, list(feature_space), name="data")
        _require_columns(df, [target_col], name="data")

        X_full = df.loc[:, feature_space]
        y_full = df.loc[:, target_col]

        # Ensure y is 1D and usable for stratification
        if stratify:
            # sklearn expects array-like with at least 2 classes for stratify
            unique = pd.Series(y_full).dropna().unique()
            if len(unique) < 2:
                raise ValueError(
                    f"Cannot stratify split: target '{target_col}' has <2 unique values: {unique}"
                )

        stratify_arg = y_full if stratify else None

        # Split FIRST, then fit scaler ONLY on train to avoid leakage
        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=test_size,
            stratify=stratify_arg,
            random_state=random_state,
        )

        # Normalize scale flag
        scale_flag = scale if isinstance(scale, bool) else str(scale).strip().lower() in {"yes", "true", "1"}

        scaler: Optional[MinMaxScaler] = None
        if scale_flag:
            scaler = MinMaxScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=X_train.columns,
                index=X_train.index,
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=X_test.columns,
                index=X_test.index,
            )
            X_train, X_test = X_train_scaled, X_test_scaled

        return SplitResult(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            X_full=X_full,
            y_full=y_full,
            scaler=scaler,
        )

    except Exception as e:
        return handle_error("Train/test split failed", e)


# In[ ]:


def _normalize_subject_reference(sr: pd.Series) -> pd.Series:
    """Ensure subject_reference has 'Patient/' prefix."""
    sr = sr.astype(str)
    return np.where(sr.str.startswith("Patient/"), sr, "Patient/" + sr)


def _compile_prefix_regex(prefixes: Sequence[str]) -> re.Pattern:
    """
    Compile a regex that matches any ICD prefix at start of string, allowing
    an optional dot or end-of-string after the prefix.
    Example prefix 'C91' matches 'C91', 'C91.0', 'C910' (depending on your code format).
    """
    escaped = "|".join(re.escape(p) for p in prefixes)
    return re.compile(rf"^(?:{escaped})(?:\.|$)")


def _categorize_conditions_any(
    codes: Sequence[str],
    *,
    rx_leuk: re.Pattern,
    rx_hemat: re.Pattern,
    rx_solid: re.Pattern,
) -> str:
    """
    Categorize a list of ICD codes using priority:
      leuk > hemat > solid > something

    Mirrors your original logic but is faster (compiled regex) and testable.
    """
    leuk = False
    hemat = False
    solid = False

    for c in codes:
        s = str(c)
        if not solid and rx_solid.match(s):
            solid = True
        if not leuk and rx_leuk.match(s):
            leuk = True
        if not hemat and rx_hemat.match(s):
            hemat = True

        # Early exit if highest priority found
        if leuk:
            return "leuk"

    if hemat:
        return "hemat"
    if solid:
        return "solid"
    return "something"


def add_static_covariates(
    data_ml: pd.DataFrame,
    *,
    feature_space: Sequence[str],
    df_conditions: pd.DataFrame,
    df_bd: pd.DataFrame,
    fever_threshold: float = 38.0,
    min_age_years: int = 18,
    copy: bool = True,
    strict: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Add static covariates (condition category, sex, age, length of stay proxy) and
    create boolean target `fever`, using subject-level condition scanning.

    Returns
    -------
    df_subset:
        Data restricted to meta columns + feature_space, deduplicated (excluding age).
    enriched:
        Full enriched dataframe (your original returned `data_ml_cond_age_sex`).
    """
    def handle_error(msg: str, exc: Exception) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if strict:
            raise
        LOGGER.warning("%s: %s", msg, exc, exc_info=True)
        empty = pd.DataFrame()
        return empty, empty

    if copy:
        data = data_ml.copy()
        conds = df_conditions.copy()
        bd = df_bd.copy()
    else:
        data, conds, bd = data_ml, df_conditions, df_bd

    # ---- Validate required columns (match your original list) ----
    try:
        _require_columns(
            data,
            ["encounter_id", "subject_reference", "time_lag_1", "recorded_time", "time_lag_target", "time_lag_target_bt_max"],
            name="data_ml",
        )
        # NOTE: your original code validated encounter_id in df_conditions, but then used subject_reference.
        # Here we validate what we actually use.
        _require_columns(conds, ["subject_reference", "conditions"], name="df_conditions")
        _require_columns(bd, ["subject_reference", "birth_date", "sex"], name="df_bd")
    except Exception as e:
        return handle_error("Schema validation failed", e)

    # ---- Normalize datetimes + booleans ----
    try:
        data["recorded_time"] = pd.to_datetime(data["recorded_time"], utc=True, errors="coerce")
        bd["birth_date"] = pd.to_datetime(bd["birth_date"], utc=True, errors="coerce")
        data["time_lag_target"] = data["time_lag_target"].astype(bool)
        data["time_lag_1"] = data["time_lag_1"].astype(bool)
    except Exception as e:
        return handle_error("Datetime/boolean normalization failed", e)

    # ---- ICD-10 prefix lists (kept from your code; consider externalizing to constants) ----
    leuk_codes = ("C91", "C92", "C93", "C94", "C95")
    hemat_codes = ("C81", "C82", "C83", "C84", "C85", "C86", "C88", "C90", "C96")
    solid_codes = (
        "C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08", "C09", "C10", "C11", "C12", "C13", "C14",
        "C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C30", "C31", "C32",
        "C33", "C34", "C37", "C38", "C39", "C40", "C41", "C43", "C44", "C45", "C46", "C47", "C48", "C49",
        "C50", "C51", "C52", "C53", "C54", "C55", "C56", "C57", "C58",
        "C60", "C61", "C62", "C63", "C64", "C65", "C66", "C67", "C68",
        "C69", "C70", "C71", "C72", "C73", "C74", "C75", "C76", "C77", "C78", "C79", "C80",
    )

    rx_leuk = _compile_prefix_regex(leuk_codes)
    rx_hemat = _compile_prefix_regex(hemat_codes)
    rx_solid = _compile_prefix_regex(solid_codes)

    # ---- Prepare and categorize conditions per subject_reference ----
    try:
        conds_small = conds[["subject_reference", "conditions"]].dropna().copy()
        grouped = conds_small.groupby("subject_reference")["conditions"].apply(list)

        cond_category = grouped.apply(
            lambda lst: _categorize_conditions_any(
                lst,
                rx_leuk=rx_leuk,
                rx_hemat=rx_hemat,
                rx_solid=rx_solid,
            )
        ).rename("cond_category")

        data = data.merge(cond_category, left_on="subject_reference", right_index=True, how="left")
        data["cond_category"] = data["cond_category"].fillna("something")

        data = pd.get_dummies(data, columns=["cond_category"], prefix="cond")
    except Exception as e:
        return handle_error("Condition categorization/merge failed", e)

    # ---- Normalize subject_reference and merge demographics ----
    try:
        data["subject_reference"] = _normalize_subject_reference(data["subject_reference"])
        bd["subject_reference"] = _normalize_subject_reference(bd["subject_reference"])

        data = data.merge(bd[["subject_reference", "birth_date", "sex"]], on="subject_reference", how="left")
        data = pd.get_dummies(data, columns=["sex"], prefix="sex")
    except Exception as e:
        return handle_error("Demographics merge/encoding failed", e)

    # ---- Age: compute age at encounter start, filter adults, then optional per-row age ----
    try:
        min_rt = data.groupby("encounter_id")["recorded_time"].min().rename("recorded_time_min")
        data = data.merge(min_rt, on="encounter_id", how="left")

        age_at_start = np.floor((data["recorded_time_min"] - data["birth_date"]).dt.days / 365.25)
        data["age_at_start"] = age_at_start

        data = data.loc[data["age_at_start"] >= min_age_years].copy()

        # keep your original per-row age recomputation
        data["age"] = np.floor((data["recorded_time"] - data["birth_date"]).dt.days / 365.25)
    except Exception as e:
        return handle_error("Age computation/filtering failed", e)

    # ---- Target fever label ----
    try:
        data["fever"] = (pd.to_numeric(data["time_lag_target_bt_max"], errors="coerce") >= fever_threshold).astype(bool)
    except Exception as e:
        return handle_error("Fever target creation failed", e)

    # ---- Length of stay proxy (same logic as your code) ----
    try:
        min_non_target = data.loc[~data["time_lag_target"]].groupby("encounter_id")["recorded_time"].min()
        min_target = data.loc[data["time_lag_target"]].groupby("encounter_id")["recorded_time"].min()
        length_stay = (min_target - min_non_target).dt.days.rename("length_stay")
        data = data.merge(length_stay, on="encounter_id", how="left")
    except Exception as e:
        return handle_error("Length-of-stay computation failed", e)

    # ---- Final subset and deduplicate excluding age ----
    try:
        meta_columns = ["encounter_id", "subject_reference", "fever"]
        columns_include = meta_columns + list(feature_space)

        missing = [c for c in columns_include if c not in data.columns]
        if missing:
            raise KeyError(f"Requested output columns missing after processing: {missing}")

        df_subset = data[columns_include].copy()

        subset_cols = [c for c in df_subset.columns if c != "age"]
        df_subset = df_subset.drop_duplicates(subset=subset_cols)

        return df_subset, data
    except Exception as e:
        return handle_error("Final subsetting/deduplication failed", e)

