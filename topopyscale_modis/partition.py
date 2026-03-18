"""Hourly rain/snow partition using TopoSCALE-interpolated temperature.

Three modes:
  1. Multi-threshold (original): fixed thresholds [-1, 0, 1, 1.5, 2] C
  2. Jennings adaptive (legacy): T_seuil = 5.7 - 0.046 x RH(%) per pixel x day
     Uses ERA5-Land surface RH (not elevation-corrected).
  3. Jennings logistic (recommended): full logistic regression from Jennings et al.
     (2018, Supplementary Table 2, doi:10.1038/s41467-018-03629-7).
     Uses TopoSCALE RH and P at pixel elevation.
     Bivariate:  p(snow) = 1 / (1 + exp(-10.04 + 1.41*Ts + 0.09*RH))
     Trivariate: p(snow) = 1 / (1 + exp(-12.80 + 1.41*Ts + 0.09*RH + 0.03*Ps))

Usage:
    # Fixed thresholds:
    era5-partition --temperature-csv temp.csv --era5land-csv precip.csv --out-csv out.csv

    # Jennings adaptive (legacy, requires dewpoint CSV):
    era5-partition --temperature-csv temp.csv --era5land-csv precip.csv \\
        --dewpoint-csv dew.csv --jennings --out-csv out.csv

    # Jennings logistic (requires TopoSCALE CSV with RH_pixel_pct):
    era5-partition --temperature-csv toposcale.csv --era5land-csv precip.csv \\
        --jennings-logistic --out-csv out.csv

    python -m topopyscale_modis.partition ...
    from topopyscale_modis.partition import partition_rain_snow, partition_jennings_logistic
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from topopyscale_modis.config import DEFAULT_THRESHOLDS_C

# Jennings et al. (2018) T50 linear regression coefficients (legacy)
JENNINGS_INTERCEPT = 5.7
JENNINGS_SLOPE = -0.046

# Jennings et al. (2018) logistic regression coefficients
# Source: Supplementary Table 2, verified against Dryad R code
# (jennings_et_al_2018_file7_precipphase_phasemethods_code.R, lines 127-131)
# and 5 independent implementations (TopoPyScale, NOAA-OWP, pysnow, etc.)
JENNINGS_LOGISTIC_BI = {"alpha": -10.04, "beta": 1.41, "gamma": 0.09}
JENNINGS_LOGISTIC_TRI = {"alpha": -12.80, "beta": 1.41, "gamma": 0.09, "lam": 0.03}


def _magnus_rh(dewpoint_k: float, temperature_k: float) -> float:
    """Compute RH(%) from dewpoint and temperature (both in Kelvin) using Magnus formula."""
    td_c = dewpoint_k - 273.15
    t_c = temperature_k - 273.15
    if t_c == td_c:
        return 100.0
    return 100.0 * math.exp(17.625 * td_c / (243.04 + td_c)) / math.exp(17.625 * t_c / (243.04 + t_c))


def partition_rain_snow(
    temperature_df: pd.DataFrame,
    precip_df: pd.DataFrame,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Partition hourly precipitation into rain/snow using multi-threshold approach.

    Parameters
    ----------
    temperature_df : DataFrame
        Columns: pixel_id, date, hour, T_pixel_C (from toposcale.py)
    precip_df : DataFrame
        Columns: pixel_id, date, hour, total_precip_m
        Hourly total precipitation in meters water equivalent.
    thresholds : list[float], optional
        Temperature thresholds in Celsius. Default: [-1, 0, 1, 1.5, 2]

    Returns
    -------
    DataFrame
        Columns: pixel_id, date, threshold_C, snow_corrected_m, rain_corrected_m
        Daily aggregated values for each threshold.
    """
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS_C

    # Detect if precip is daily (no 'hour' column) or hourly
    precip_is_daily = "hour" not in precip_df.columns

    if precip_is_daily:
        # ERA5-Land precip is daily -> use hourly T to compute snow_fraction per day
        # snow_fraction = fraction of hours in the day where T < threshold
        # Then: snow_m = snow_fraction x total_precip_m (daily)
        print("[partition] ERA5-Land precip is daily -- using hourly T snow_fraction approach")

        results = []
        for thresh in thresholds:
            # Compute daily snow fraction from hourly temperature
            temp = temperature_df.copy()
            temp["is_snow"] = (temp["T_pixel_C"] < thresh).astype(float)
            snow_frac = (
                temp.groupby(["pixel_id", "date"], as_index=False)
                .agg(snow_fraction=("is_snow", "mean"))
            )

            # Merge with daily precip
            daily = pd.merge(snow_frac, precip_df[["pixel_id", "date", "total_precip_m"]],
                             on=["pixel_id", "date"], how="inner")

            if daily.empty:
                print(f"[partition] WARNING: no matching records for threshold {thresh} C")
                continue

            daily["snow_corrected_m"] = daily["snow_fraction"] * daily["total_precip_m"]
            daily["rain_corrected_m"] = (1.0 - daily["snow_fraction"]) * daily["total_precip_m"]
            daily["threshold_C"] = thresh
            daily = daily[["pixel_id", "date", "threshold_C", "snow_corrected_m", "rain_corrected_m"]]
            results.append(daily)

    else:
        # Hourly precip -- original approach
        merge_keys = ["pixel_id", "date", "hour"]
        hourly = pd.merge(temperature_df, precip_df, on=merge_keys, how="inner")

        if hourly.empty:
            print("[partition] WARNING: no matching records after merge -- check data alignment")
            return pd.DataFrame(columns=["pixel_id", "date", "threshold_C", "snow_corrected_m", "rain_corrected_m"])

        results = []
        for thresh in thresholds:
            is_snow = hourly["T_pixel_C"] < thresh
            df_thresh = hourly.copy()
            df_thresh["snow_m"] = df_thresh["total_precip_m"].where(is_snow, 0.0)
            df_thresh["rain_m"] = df_thresh["total_precip_m"].where(~is_snow, 0.0)

            daily = (
                df_thresh.groupby(["pixel_id", "date"], as_index=False)
                .agg(snow_corrected_m=("snow_m", "sum"), rain_corrected_m=("rain_m", "sum"))
            )
            daily["threshold_C"] = thresh
            results.append(daily)

    out = pd.concat(results, ignore_index=True)
    out = out.sort_values(["pixel_id", "date", "threshold_C"]).reset_index(drop=True)

    n_pixels = out["pixel_id"].nunique()
    n_days = out["date"].nunique()
    print(f"[partition] {len(out)} rows ({n_pixels} pixels x {n_days} days x {len(thresholds)} thresholds)")
    return out


def partition_jennings(
    temperature_df: pd.DataFrame,
    precip_df: pd.DataFrame,
    dewpoint_df: pd.DataFrame,
) -> pd.DataFrame:
    """Partition precipitation using Jennings (2018) RH-adaptive threshold.

    T_seuil(pixel, day) = 5.7 - 0.046 x RH(%)
    where RH is computed from ERA5-Land T2m and dewpoint (Magnus formula).

    The threshold is compared against TopoSCALE T_pixel (hourly if available).

    Parameters
    ----------
    temperature_df : DataFrame
        Columns: pixel_id, date, hour, T_pixel_C (from toposcale.py)
    precip_df : DataFrame
        Columns: pixel_id, date, total_precip_m, temperature_2m_K
        Daily ERA5-Land precipitation with T2m for RH computation.
    dewpoint_df : DataFrame
        Columns: pixel_id, date, dewpoint_2m_K
        Daily ERA5-Land dewpoint temperature.

    Returns
    -------
    DataFrame
        Columns: pixel_id, date, threshold_C, rh_pct,
                 snow_corrected_m, rain_corrected_m
    """
    # Step 1: compute daily RH and Jennings threshold from ERA5-Land T2m + dewpoint
    dew_daily = dewpoint_df[["pixel_id", "date", "dewpoint_2m_K"]].copy()

    # Get T2m from precip table
    precip_cols = ["pixel_id", "date", "total_precip_m", "temperature_2m_K"]
    precip_daily = precip_df[precip_cols].copy()

    rh_df = pd.merge(precip_daily, dew_daily, on=["pixel_id", "date"], how="inner")

    # Vectorized Magnus RH
    td_c = rh_df["dewpoint_2m_K"] - 273.15
    t_c = rh_df["temperature_2m_K"] - 273.15
    rh_df["rh_pct"] = (
        100.0 * np.exp(17.625 * td_c / (243.04 + td_c))
        / np.exp(17.625 * t_c / (243.04 + t_c))
    ).clip(0, 100)
    rh_df["threshold_C"] = JENNINGS_INTERCEPT + JENNINGS_SLOPE * rh_df["rh_pct"]

    print(f"[partition/jennings] RH: mean={rh_df['rh_pct'].mean():.1f}%, "
          f"p10={rh_df['rh_pct'].quantile(0.1):.1f}%, p90={rh_df['rh_pct'].quantile(0.9):.1f}%")
    print(f"[partition/jennings] Threshold: mean={rh_df['threshold_C'].mean():.2f} C, "
          f"min={rh_df['threshold_C'].min():.2f} C, max={rh_df['threshold_C'].max():.2f} C")

    # Step 2: apply pixel x day threshold to hourly TopoSCALE temperature
    precip_is_daily = "hour" not in temperature_df.columns

    # Merge threshold onto temperature data
    thresh_cols = rh_df[["pixel_id", "date", "threshold_C", "rh_pct", "total_precip_m"]].copy()

    if precip_is_daily:
        # Daily T -> use daily mean T_pixel_C
        if "T_pixel_C" not in temperature_df.columns:
            # temperature_df might have hourly data after all; aggregate
            temp_daily = (
                temperature_df.groupby(["pixel_id", "date"], as_index=False)
                .agg(T_pixel_C=("T_pixel_C", "mean"))
            )
        else:
            temp_daily = temperature_df[["pixel_id", "date", "T_pixel_C"]].copy()

        merged = pd.merge(temp_daily, thresh_cols, on=["pixel_id", "date"], how="inner")
        is_snow = merged["T_pixel_C"] < merged["threshold_C"]
        merged["snow_corrected_m"] = merged["total_precip_m"].where(is_snow, 0.0)
        merged["rain_corrected_m"] = merged["total_precip_m"].where(~is_snow, 0.0)

    else:
        # Hourly T -> snow_fraction approach with adaptive threshold
        hourly = pd.merge(
            temperature_df[["pixel_id", "date", "hour", "T_pixel_C"]],
            thresh_cols,
            on=["pixel_id", "date"],
            how="inner",
        )
        hourly["is_snow"] = (hourly["T_pixel_C"] < hourly["threshold_C"]).astype(float)
        merged = (
            hourly.groupby(["pixel_id", "date"], as_index=False)
            .agg(
                snow_fraction=("is_snow", "mean"),
                threshold_C=("threshold_C", "first"),
                rh_pct=("rh_pct", "first"),
                total_precip_m=("total_precip_m", "first"),
            )
        )
        merged["snow_corrected_m"] = merged["snow_fraction"] * merged["total_precip_m"]
        merged["rain_corrected_m"] = (1.0 - merged["snow_fraction"]) * merged["total_precip_m"]

    out = merged[["pixel_id", "date", "threshold_C", "rh_pct", "snow_corrected_m", "rain_corrected_m"]].copy()
    out = out.sort_values(["pixel_id", "date"]).reset_index(drop=True)

    n_pixels = out["pixel_id"].nunique()
    n_days = out["date"].nunique()
    print(f"[partition/jennings] {len(out)} rows ({n_pixels} pixels x {n_days} days)")
    return out


def _jennings_p_snow(t_c: np.ndarray, rh_pct: np.ndarray,
                     p_hpa: np.ndarray | None = None) -> np.ndarray:
    """Compute p(snow) using Jennings (2018) logistic regression.

    Bivariate (T + RH) if p_hpa is None, trivariate (T + RH + P) otherwise.
    Coefficients from Supplementary Table 2 (Dryad file7 R code, lines 127-131).

    Parameters
    ----------
    t_c : array  -- air temperature in C
    rh_pct : array  -- relative humidity in %
    p_hpa : array or None  -- surface pressure in hPa (millibars).
        Internally converted to kPa (/ 10) to match the Jennings model units
        (paper bins: 60-70, 70-80, 80-90, 90-105 kPa; TopoPyScale uses sp/1000).

    Returns
    -------
    array of p(snow) in [0, 1]
    """
    if p_hpa is not None:
        c = JENNINGS_LOGISTIC_TRI
        p_kpa = p_hpa / 10.0  # hPa -> kPa
        logit = c["alpha"] + c["beta"] * t_c + c["gamma"] * rh_pct + c["lam"] * p_kpa
    else:
        c = JENNINGS_LOGISTIC_BI
        logit = c["alpha"] + c["beta"] * t_c + c["gamma"] * rh_pct
    return 1.0 / (1.0 + np.exp(logit))


def partition_jennings_logistic(
    temperature_df: pd.DataFrame,
    precip_df: pd.DataFrame,
    use_trivariate: bool = True,
) -> pd.DataFrame:
    """Partition precipitation using Jennings (2018) full logistic model.

    Uses TopoSCALE-downscaled RH and P at pixel elevation (not ERA5-Land surface).
    This eliminates the blocky T50 artifacts from the static Jennings raster.

    The temperature_df must come from toposcale.py with humidity enabled,
    providing columns: T_pixel_C, RH_pixel_pct, and optionally P_pixel_hPa.

    Parameters
    ----------
    temperature_df : DataFrame
        Columns: pixel_id, date, hour, T_pixel_C, RH_pixel_pct, P_pixel_hPa
        (from toposcale.py with humidity interpolation)
    precip_df : DataFrame
        Columns: pixel_id, date, total_precip_m
        Daily ERA5-Land precipitation.
    use_trivariate : bool
        If True and P_pixel_hPa is available, use the trivariate model (T+RH+P).
        Otherwise fall back to bivariate (T+RH).

    Returns
    -------
    DataFrame
        Columns: pixel_id, date, p_snow_mean, snow_corrected_m, rain_corrected_m
    """
    # Validate required columns
    required = {"pixel_id", "date", "T_pixel_C", "RH_pixel_pct"}
    missing = required - set(temperature_df.columns)
    if missing:
        raise ValueError(
            f"temperature_df missing columns {missing}. "
            "Run toposcale.py with humidity (--era5-nc must contain specific_humidity)."
        )

    has_pressure = "P_pixel_hPa" in temperature_df.columns
    model_name = "trivariate" if (use_trivariate and has_pressure) else "bivariate"
    print(f"[partition/jennings-logistic] Using {model_name} model")

    # Compute p(snow) for each hourly observation
    t_c = temperature_df["T_pixel_C"].values
    rh = temperature_df["RH_pixel_pct"].values
    p_hpa = temperature_df["P_pixel_hPa"].values if (use_trivariate and has_pressure) else None

    p_snow = _jennings_p_snow(t_c, rh, p_hpa)

    temp = temperature_df[["pixel_id", "date"]].copy()
    temp["p_snow"] = p_snow

    # Aggregate to daily: mean p(snow) across hours = snow fraction
    daily_psnow = (
        temp.groupby(["pixel_id", "date"], as_index=False)
        .agg(p_snow_mean=("p_snow", "mean"))
    )

    # Merge with daily precipitation
    precip_cols = [c for c in ["pixel_id", "date", "total_precip_m"] if c in precip_df.columns]
    merged = pd.merge(daily_psnow, precip_df[precip_cols], on=["pixel_id", "date"], how="inner")

    if merged.empty:
        print("[partition/jennings-logistic] WARNING: no matching records after merge")
        return pd.DataFrame(columns=["pixel_id", "date", "p_snow_mean",
                                     "snow_corrected_m", "rain_corrected_m"])

    merged["snow_corrected_m"] = merged["p_snow_mean"] * merged["total_precip_m"]
    merged["rain_corrected_m"] = (1.0 - merged["p_snow_mean"]) * merged["total_precip_m"]

    out = merged[["pixel_id", "date", "p_snow_mean", "snow_corrected_m", "rain_corrected_m"]].copy()
    out = out.sort_values(["pixel_id", "date"]).reset_index(drop=True)

    n_pixels = out["pixel_id"].nunique()
    n_days = out["date"].nunique()
    print(f"[partition/jennings-logistic] {len(out)} rows ({n_pixels} pixels x {n_days} days)")
    print(f"[partition/jennings-logistic] p(snow): mean={out['p_snow_mean'].mean():.3f}, "
          f"p10={out['p_snow_mean'].quantile(0.1):.3f}, p90={out['p_snow_mean'].quantile(0.9):.3f}")
    return out


def partition_jennings_logistic_db(
    db_path: Path,
    use_trivariate: bool = True,
    table_name: str = "era5.snow_partition_jennings_logistic",
) -> None:
    """Compute Jennings logistic snow partition directly in DuckDB.

    Reads from era5.temperature_at_pixel (TopoSCALE with humidity) and
    era5.precipitation_era5land, computes p(snow) per pixel x day, and
    writes results to a new table.

    This is faster than the CSV approach and avoids the roundtrip.
    """
    import duckdb

    con = duckdb.connect(str(db_path))

    # Verify required tables exist
    tables = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'era5'"
        ).fetchall()
    }
    if "temperature_at_pixel" not in tables:
        raise ValueError("era5.temperature_at_pixel not found -- run ingest first")
    if "precipitation_era5land" not in tables:
        raise ValueError("era5.precipitation_era5land not found -- run ingest first")

    # Check humidity columns exist
    temp_cols = {
        row[0]
        for row in con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'era5' AND table_name = 'temperature_at_pixel'"
        ).fetchall()
    }
    if "rh_pixel_pct" not in temp_cols:
        raise ValueError(
            "rh_pixel_pct not found in era5.temperature_at_pixel -- "
            "TopoSCALE must be run with humidity (NetCDF must contain specific_humidity)"
        )

    has_pressure = "p_pixel_hpa" in temp_cols
    model_name = "trivariate" if (use_trivariate and has_pressure) else "bivariate"

    if use_trivariate and has_pressure:
        c = JENNINGS_LOGISTIC_TRI
        logit_expr = (
            f"{c['alpha']} + {c['beta']} * t.t_pixel_c "
            f"+ {c['gamma']} * t.rh_pixel_pct "
            f"+ {c['lam']} * (t.p_pixel_hpa / 10.0)"
        )
    else:
        c = JENNINGS_LOGISTIC_BI
        logit_expr = (
            f"{c['alpha']} + {c['beta']} * t.t_pixel_c "
            f"+ {c['gamma']} * t.rh_pixel_pct"
        )

    print(f"[partition/jennings-logistic-db] Using {model_name} model")
    print(f"[partition/jennings-logistic-db] logit = {logit_expr}")

    # Compute p(snow) per hour, aggregate to daily mean, join with precip
    sql = f"""
        CREATE OR REPLACE TABLE {table_name} AS
        WITH hourly_psnow AS (
            SELECT
                pixel_id,
                date,
                1.0 / (1.0 + exp({logit_expr})) AS p_snow
            FROM era5.temperature_at_pixel t
            WHERE t.rh_pixel_pct IS NOT NULL
        ),
        daily_psnow AS (
            SELECT
                pixel_id,
                date,
                avg(p_snow) AS p_snow_mean
            FROM hourly_psnow
            GROUP BY pixel_id, date
        )
        SELECT
            d.pixel_id,
            d.date,
            round(d.p_snow_mean, 4) AS p_snow_mean,
            round(d.p_snow_mean * p.total_precip_m, 8) AS snow_corrected_m,
            round((1.0 - d.p_snow_mean) * p.total_precip_m, 8) AS rain_corrected_m
        FROM daily_psnow d
        INNER JOIN era5.precipitation_era5land p
            ON d.pixel_id = p.pixel_id AND d.date = p.date
        ORDER BY d.pixel_id, d.date
    """
    con.execute(sql)

    # Report stats
    stats = con.execute(f"""
        SELECT
            count(*) AS n_rows,
            count(DISTINCT pixel_id) AS n_pixels,
            count(DISTINCT date) AS n_days,
            avg(p_snow_mean) AS mean_psnow,
            percentile_cont(0.1) WITHIN GROUP (ORDER BY p_snow_mean) AS p10,
            percentile_cont(0.9) WITHIN GROUP (ORDER BY p_snow_mean) AS p90
        FROM {table_name}
    """).fetchone()
    print(f"[partition/jennings-logistic-db] {stats[0]} rows ({stats[1]} pixels x {stats[2]} days)")
    print(f"[partition/jennings-logistic-db] p(snow): mean={stats[3]:.3f}, p10={stats[4]:.3f}, p90={stats[5]:.3f}")

    # Update the cross-schema view
    from topopyscale_modis.ingest import _create_obs_with_era5_view
    _create_obs_with_era5_view(con)

    con.close()
    print(f"[partition/jennings-logistic-db] Done: {db_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Partition hourly precipitation into rain/snow using TopoSCALE temperature",
    )

    # DuckDB mode (recommended)
    parser.add_argument(
        "--db", type=Path, default=None,
        help="DuckDB path -- compute partition directly from ingested ERA5 tables",
    )
    parser.add_argument(
        "--method",
        choices=["jennings-logistic", "jennings", "multi-threshold"],
        default=None,
        help="Partition method (with --db mode)",
    )

    # CSV mode (legacy)
    parser.add_argument(
        "--temperature-csv",
        type=Path,
        default=None,
        help="CSV from era5-interpolate (pixel_id, date, hour, T_pixel_C)",
    )
    parser.add_argument(
        "--era5land-csv",
        type=Path,
        default=None,
        help="CSV with ERA5-Land precipitation (pixel_id, date, total_precip_m, temperature_2m_K)",
    )
    parser.add_argument(
        "--dewpoint-csv",
        type=Path,
        default=None,
        help="ERA5-Land dewpoint CSV (pixel_id, date, dewpoint_2m_K). Required for --jennings.",
    )
    parser.add_argument(
        "--jennings",
        action="store_true",
        help="Use Jennings (2018) RH-adaptive threshold (legacy, requires --dewpoint-csv)",
    )
    parser.add_argument(
        "--jennings-logistic",
        action="store_true",
        help="Use Jennings (2018) full logistic model (recommended, requires TopoSCALE CSV with RH_pixel_pct)",
    )
    parser.add_argument(
        "--bivariate",
        action="store_true",
        help="Force bivariate model (T+RH only) instead of trivariate (T+RH+P)",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default=",".join(str(t) for t in DEFAULT_THRESHOLDS_C),
        help="Comma-separated temperature thresholds in Celsius (default: -1,0,1,1.5,2)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Output CSV path (required for CSV mode)",
    )
    args = parser.parse_args()

    # DuckDB mode
    if args.db is not None:
        method = args.method or "jennings-logistic"
        if method == "jennings-logistic":
            partition_jennings_logistic_db(
                args.db,
                use_trivariate=not args.bivariate,
            )
        else:
            parser.error(f"--db mode only supports 'jennings-logistic' (got '{method}')")
        return

    # CSV mode (legacy)
    if args.temperature_csv is None or args.era5land_csv is None:
        parser.error("--temperature-csv and --era5land-csv required (or use --db mode)")
    if args.out_csv is None:
        parser.error("--out-csv required for CSV mode")

    temp_df = pd.read_csv(args.temperature_csv)
    precip_df = pd.read_csv(args.era5land_csv)

    if args.jennings_logistic:
        out = partition_jennings_logistic(temp_df, precip_df,
                                         use_trivariate=not args.bivariate)
    elif args.jennings:
        if args.dewpoint_csv is None:
            parser.error("--dewpoint-csv is required when using --jennings")
        dew_df = pd.read_csv(args.dewpoint_csv)
        out = partition_jennings(temp_df, precip_df, dew_df)
    else:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]
        out = partition_rain_snow(temp_df, precip_df, thresholds)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"[partition] Wrote {args.out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
