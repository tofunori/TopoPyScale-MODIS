"""TopoSCALE interpolation (Fiddes & Gruber 2014).

Interpolates ERA5 pressure-level temperature (and optionally specific humidity)
to glacier pixel elevations using the two bracketing pressure levels.

When specific_humidity is available in the NetCDF, also computes:
  - P_pixel (hPa): pixel pressure from log-linear interpolation
  - q_pixel (kg/kg): interpolated specific humidity
  - Td_pixel (C): dewpoint from q and P via Magnus inverse
  - RH_pixel (%): relative humidity from Td and T via Magnus formula

Usage:
    era5-interpolate --era5-nc /db/era5/file.nc --db modis.duckdb --region 02-03 --out-csv out.csv
    era5-interpolate --era5-nc /db/era5/file.nc --pixels-csv pixels.csv --out-csv out.csv --workers 20
    python -m topopyscale_modis.toposcale ...
    from topopyscale_modis.toposcale import interpolate_temperature
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from topopyscale_modis.config import G, normalize_region, validate_region


def load_pixel_elevations(db_path: Path, region: str) -> pd.DataFrame:
    """Load pixel_id + elevation from the DuckDB modis.pixel_static table.

    Returns DataFrame with columns: pixel_id, elevation, latitude, longitude.
    """
    import duckdb  # lazy: only needed at runtime

    region_us = normalize_region(region)
    con = duckdb.connect(str(db_path), read_only=True)
    df = con.execute(
        """
        SELECT DISTINCT pixel_id, elevation, latitude, longitude
        FROM modis.pixel_static
        WHERE region = ? AND elevation IS NOT NULL
        """,
        [region_us],
    ).fetchdf()
    con.close()

    if df.empty:
        raise ValueError(f"No pixels with elevation found for region {region_us!r}")
    print(f"[toposcale] Loaded {len(df)} pixels from {db_path.name}")
    return df


def _process_pixel_chunk(args: tuple) -> list:
    """Process a subset of pixels -- called by multiprocessing workers.

    Each worker opens the NetCDF independently to avoid pickling issues.
    Returns a list of result dicts.
    """
    import numpy as np
    import pandas as pd
    import xarray as xr

    era5_nc, pixel_chunk, g, cds_format, has_humidity, worker_id = args

    ds = xr.open_dataset(era5_nc, engine="h5netcdf")
    t_var = "t" if "t" in ds else "temperature"
    z_var = "z" if "z" in ds else "geopotential"
    q_var = "q" if "q" in ds else "specific_humidity" if "specific_humidity" in ds else None

    # Override flag if variable not actually in file
    if q_var is None:
        has_humidity = False

    if cds_format:
        time_coord = ds["valid_time"]
    else:
        time_coord = ds["time"]

    era5_lats = ds["latitude"].values
    era5_lons = ds["longitude"].values
    times = pd.to_datetime(time_coord.values)
    lev_name = "level" if "level" in ds else "pressure_level"
    pressure_levels = ds[lev_name].values.astype(float)  # hPa

    results = []

    for i, (_, px) in enumerate(pixel_chunk.iterrows()):
        if worker_id == 0 and i % 100 == 0:
            print(f"[toposcale] worker-0: {i}/{len(pixel_chunk)} pixels", flush=True)

        z_pixel = px["elevation"]
        px_lat = px["latitude"]
        px_lon = px["longitude"]

        # IDW 3x3 patch (Fiddes & Gruber 2014) -- inverse-distance weighted
        # average of the 9 surrounding ERA5 grid cells.
        lat_idx = int(np.argmin(np.abs(era5_lats - px_lat)))
        lon_idx = int(np.argmin(np.abs(era5_lons - px_lon)))
        lat_lo = max(0, lat_idx - 1)
        lat_hi = min(len(era5_lats) - 1, lat_idx + 1) + 1  # +1 for slice
        lon_lo = max(0, lon_idx - 1)
        lon_hi = min(len(era5_lons) - 1, lon_idx + 1) + 1

        patch_lats = era5_lats[lat_lo:lat_hi]
        patch_lons = era5_lons[lon_lo:lon_hi]
        lat_grid, lon_grid = np.meshgrid(patch_lats, patch_lons, indexing="ij")
        dist = np.sqrt((px_lat - lat_grid) ** 2 + (px_lon - lon_grid) ** 2)
        dist = np.maximum(dist, 1e-10)  # avoid div-by-zero at exact grid point
        idw = 1.0 / dist ** 2
        w = idw / idw.sum()  # shape [nlat, nlon]

        if cds_format:
            t_raw = ds[t_var][:, :, :, lat_lo:lat_hi, lon_lo:lon_hi].values
            z_raw = ds[z_var][:, :, :, lat_lo:lat_hi, lon_lo:lon_hi].values / g
            t_profile = np.nanmean(np.einsum("tvlij,ij->tvl", t_raw, w), axis=0)
            z_profile = np.nanmean(np.einsum("tvlij,ij->tvl", z_raw, w), axis=0)
            if has_humidity:
                q_raw = ds[q_var][:, :, :, lat_lo:lat_hi, lon_lo:lon_hi].values
                q_profile = np.nanmean(np.einsum("tvlij,ij->tvl", q_raw, w), axis=0)
        else:
            t_patch = ds[t_var][:, :, lat_lo:lat_hi, lon_lo:lon_hi].values
            z_patch = ds[z_var][:, :, lat_lo:lat_hi, lon_lo:lon_hi].values / g
            t_profile = np.einsum("tlij,ij->tl", t_patch, w)
            z_profile = np.einsum("tlij,ij->tl", z_patch, w)
            if has_humidity:
                q_patch = ds[q_var][:, :, lat_lo:lat_hi, lon_lo:lon_hi].values
                q_profile = np.einsum("tlij,ij->tl", q_patch, w)

        for ti, dt in enumerate(times):
            z_col = z_profile[ti, :]
            t_col = t_profile[ti, :]

            sort_idx = np.argsort(z_col)
            z_sorted = z_col[sort_idx]
            t_sorted = t_col[sort_idx]
            p_sorted = pressure_levels[sort_idx]

            if has_humidity:
                q_col = q_profile[ti, :]
                q_sorted = q_col[sort_idx]

            if z_pixel <= z_sorted[0]:
                # Fiddes & Gruber (2014): use nearest level above pixel,
                # compute pressure via barometric formula.  Relevant for
                # tidewater glaciers (01-04, 01-05, 01-06) at ~0 m.
                idx_above = int(np.searchsorted(z_sorted, z_pixel, side="right"))
                idx_above = min(idx_above, len(z_sorted) - 1)
                t_interp = t_sorted[idx_above]
                lapse = np.nan
                if has_humidity:
                    q_interp = q_sorted[idx_above]
                    R_d = 287.05  # dry air gas constant
                    t_mean = 0.5 * (t_sorted[idx_above] + t_interp)
                    p_interp = p_sorted[idx_above] * np.exp(
                        -(z_pixel - z_sorted[idx_above]) / (t_mean * R_d / g)
                    )
            elif z_pixel >= z_sorted[-1]:
                dz = z_sorted[-1] - z_sorted[-2]
                dt_lev = t_sorted[-1] - t_sorted[-2]
                lapse = (dt_lev / dz) * 1000 if dz > 0 else np.nan
                t_interp = t_sorted[-1] + (dt_lev / dz) * (z_pixel - z_sorted[-1]) if dz > 0 else t_sorted[-1]
                if has_humidity:
                    q_interp = q_sorted[-1]  # clamp q at highest level
                    # Log-linear pressure extrapolation
                    if dz > 0:
                        p_interp = p_sorted[-1] * np.exp(-g * (z_pixel - z_sorted[-1]) / (287.05 * t_interp))
                    else:
                        p_interp = p_sorted[-1]
            else:
                idx_above = int(np.searchsorted(z_sorted, z_pixel))
                idx_below = idx_above - 1
                z_lo, z_hi = z_sorted[idx_below], z_sorted[idx_above]
                t_lo, t_hi = t_sorted[idx_below], t_sorted[idx_above]
                dz = z_hi - z_lo
                frac = (z_pixel - z_lo) / dz if dz > 0 else 0.0
                t_interp = t_lo + frac * (t_hi - t_lo)
                lapse = ((t_hi - t_lo) / dz) * 1000 if dz > 0 else np.nan
                if has_humidity:
                    q_interp = q_sorted[idx_below] + frac * (q_sorted[idx_above] - q_sorted[idx_below])
                    # Barometric pressure (Fiddes & Gruber 2014): use upper
                    # bracketing level + mean temperature between level and pixel.
                    R_d = 287.05
                    t_mean = 0.5 * (t_sorted[idx_above] + t_interp)
                    p_interp = p_sorted[idx_above] * np.exp(
                        -(z_pixel - z_sorted[idx_above]) / (t_mean * R_d / g)
                    )

            row = {
                "pixel_id": int(px["pixel_id"]),
                "date": dt.strftime("%Y-%m-%d"),
                "hour": dt.hour,
                "T_pixel_K": round(float(t_interp), 3),
                "T_pixel_C": round(float(t_interp) - 273.15, 3),
                "z_pixel_m": round(float(z_pixel), 1),
                "lapse_rate_Ckm": round(float(lapse), 3) if not np.isnan(lapse) else None,
            }

            if has_humidity:
                t_c = float(t_interp) - 273.15
                # Vapor pressure from q and P: e = q * P / (0.622 + 0.378 * q)
                e = float(q_interp) * float(p_interp) / (0.622 + 0.378 * float(q_interp))
                # Dewpoint from Magnus inverse: Td = 243.04 * ln(e/6.112) / (17.625 - ln(e/6.112))
                if e > 0:
                    ln_ratio = np.log(e / 6.112)
                    td_c = 243.04 * ln_ratio / (17.625 - ln_ratio)
                else:
                    td_c = np.nan
                # RH from Magnus formula
                if not np.isnan(td_c):
                    rh = 100.0 * np.exp(17.625 * td_c / (243.04 + td_c)) / np.exp(17.625 * t_c / (243.04 + t_c))
                    rh = max(0.0, min(100.0, rh))
                else:
                    rh = np.nan

                row["q_pixel_kgkg"] = round(float(q_interp), 7)
                row["P_pixel_hPa"] = round(float(p_interp), 2)
                row["Td_pixel_C"] = round(td_c, 3) if not np.isnan(td_c) else None
                row["RH_pixel_pct"] = round(rh, 2) if not np.isnan(rh) else None

            results.append(row)

    ds.close()
    return results


def interpolate_temperature(
    era5_nc: Path,
    pixel_elevations: pd.DataFrame,
    workers: int = 1,
) -> pd.DataFrame:
    """Interpolate ERA5 pressure-level temperature (and humidity) to each pixel elevation.

    Algorithm (Fiddes & Gruber 2014):
      1. Open ERA5 NetCDF with T(time, level, lat, lon) and Z(time, level, lat, lon)
      2. Convert geopotential to altitude: z = geopotential / g
      3. For each pixel, find nearest ERA5 grid cell
      4. For each time step, find the two pressure levels bracketing the pixel altitude
      5. Linearly interpolate T (and q if available) between those two levels
      6. If q available: compute P_pixel, Td, RH at pixel altitude
      7. Handle below-surface case (pixel below lowest level) via clamping

    Args:
        workers: Number of parallel processes (default 1). Use --workers on Narval/HPC.

    Returns DataFrame with columns:
        pixel_id, date, hour, T_pixel_K, T_pixel_C, z_pixel_m, lapse_rate_Ckm
        + if humidity available: q_pixel_kgkg, P_pixel_hPa, Td_pixel_C, RH_pixel_pct
    """
    from multiprocessing import Pool

    import xarray as xr

    # Detect format once in main process
    ds = xr.open_dataset(era5_nc, engine="h5netcdf")
    has_humidity = "q" in ds or "specific_humidity" in ds
    cds_format = (
        "valid_time" in ds.coords
        and "valid_time" in ds.dims
        and "time" in ds.dims
        and len(ds.dims) >= 5
    )
    if cds_format:
        n_chunks = ds.sizes["time"]
        n_times = ds.sizes["valid_time"]
        print(f"[toposcale] CDS format: {n_chunks} chunks x {n_times} timesteps")
    if has_humidity:
        print("[toposcale] Humidity detected -> will compute q, P, Td, RH at pixel")
    ds.close()

    print(f"[toposcale] {len(pixel_elevations)} pixels, {workers} worker(s)")

    # Split pixels across workers (keep as DataFrames, not numpy arrays)
    n = len(pixel_elevations)
    chunk_size = max(1, n // workers)
    chunks = [pixel_elevations.iloc[i:i + chunk_size] for i in range(0, n, chunk_size)]
    job_args = [(era5_nc, chunk, G, cds_format, has_humidity, i) for i, chunk in enumerate(chunks)]

    if workers == 1:
        all_results = _process_pixel_chunk(job_args[0])
    else:
        with Pool(workers) as pool:
            results_per_worker = pool.map(_process_pixel_chunk, job_args)
        all_results = [row for chunk in results_per_worker for row in chunk]

    df = pd.DataFrame(all_results)
    print(f"[toposcale] Interpolated {len(df)} records ({df['pixel_id'].nunique()} pixels)")
    if has_humidity and "RH_pixel_pct" in df.columns:
        rh = df["RH_pixel_pct"].dropna()
        print(f"[toposcale] RH: mean={rh.mean():.1f}%, p10={rh.quantile(0.1):.1f}%, p90={rh.quantile(0.9):.1f}%")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TopoSCALE: interpolate ERA5 temperature to glacier pixel elevations",
    )
    parser.add_argument(
        "--era5-nc",
        type=Path,
        required=True,
        help="ERA5 pressure-level NetCDF file",
    )
    parser.add_argument(
        "--pixels-csv",
        type=Path,
        default=None,
        help="CSV with pixel_id, elevation, latitude, longitude (alternative to --db)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument("--db", type=Path, default=None, help="DuckDB path (alternative to --pixels-csv)")
    parser.add_argument("--region", type=str, default=None, help="Region code e.g. 02-03 (required with --db)")
    parser.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output CSV path",
    )
    args = parser.parse_args()

    if args.pixels_csv is not None:
        pixels = pd.read_csv(args.pixels_csv)
        print(f"[toposcale] Loaded {len(pixels)} pixels from {args.pixels_csv.name}")
    else:
        validate_region(args.region)
        pixels = load_pixel_elevations(args.db, args.region)

    df = interpolate_temperature(args.era5_nc, pixels, workers=args.workers)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[toposcale] Wrote {args.out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
