"""Download ERA5 pressure-level data (T + Z + q) via CDS API.

Downloads each month in 10-day chunks to stay within CDS cost limits
(~10 000 fields per request), then merges into one JJA NetCDF per region/year.

Also provides ``download_humidity_only()`` to retrofit specific_humidity into
existing T+Z-only NetCDF files.

Usage:
    era5-download --region 02-03 --year 2023 --months 06,07,08 --out-dir /db/era5
    era5-download --region 02-03 --year 2023 --out-dir /db/era5 --humidity-only
    python -m topopyscale_modis.fetch_era5 --region 02-03 --year 2023 --months 06,07,08
    from topopyscale_modis.fetch_era5 import download_pressure_levels
"""

from __future__ import annotations

import argparse
import calendar
import threading
from pathlib import Path

from topopyscale_modis._cli import add_region_arg, parse_months
from topopyscale_modis.config import (
    CDS_VARIABLES,
    PRESSURE_LEVELS_HPA,
    REGION_BBOX,
    validate_region,
)

_MERGE_LOCK = threading.Lock()  # serialise xarray merges to avoid OOM with parallel workers


_CHUNK_DAYS_FULL = 7   # 3 vars x 16 levels x 7d x 24h = 8064 < 10k CDS limit
_CHUNK_DAYS_Q    = 10  # 1 var  x 16 levels x 10d x 24h = 3840 < 10k CDS limit


def _all_days(year: int, month: int) -> list[str]:
    """Return all days of *month* as zero-padded strings."""
    n_days = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n_days + 1)]


def _day_chunks(year: int, month: int, chunk_days: int) -> list[list[str]]:
    """Split a month into chunks of *chunk_days* days."""
    days = _all_days(year, month)
    return [days[i : i + chunk_days] for i in range(0, len(days), chunk_days)]


def download_pressure_levels(
    region: str,
    year: int,
    months: list[int],
    out_dir: Path,
) -> Path:
    """Download ERA5 pressure-level T + Z + q from CDS.

    Each month is requested as a single CDS call (full-month chunk), then the
    monthly files are merged into one JJA NetCDF.  Skips if the final file
    already exists.

    Returns the path to the merged NetCDF file.
    """
    import cdsapi

    region = validate_region(region)
    out_dir.mkdir(parents=True, exist_ok=True)

    month_tag = "_".join(f"{m:02d}" for m in sorted(months))
    out_file = out_dir / f"era5_pressure_{region}_{year}_{month_tag}.nc"

    if out_file.exists():
        print(f"[skip] {out_file} already exists")
        return out_file

    bbox = REGION_BBOX.get(region)
    if bbox is None:
        raise ValueError(
            f"No bounding box for region {region!r}. "
            f"Add it to topopyscale_modis.config.REGION_BBOX"
        )

    client = cdsapi.Client()
    chunk_files: list[Path] = []

    for month in sorted(months):
        for ci, days in enumerate(_day_chunks(year, month, _CHUNK_DAYS_FULL)):
            chunk_file = out_dir / f"era5_pressure_{region}_{year}_{month:02d}_chunk{ci}.nc"
            chunk_files.append(chunk_file)

            if chunk_file.exists():
                print(f"[skip] {chunk_file.name} already exists")
                continue

            request = {
                "product_type": "reanalysis",
                "variable": CDS_VARIABLES,
                "pressure_level": [str(p) for p in PRESSURE_LEVELS_HPA],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": bbox,
                "data_format": "netcdf",
            }

            n_fields = len(CDS_VARIABLES) * len(PRESSURE_LEVELS_HPA) * len(days) * 24
            print(
                f"[CDS] {region}/{year}/{month:02d} chunk{ci}: "
                f"days {days[0]}-{days[-1]}, {n_fields} fields, bbox {bbox}"
            )

            client.retrieve(
                "reanalysis-era5-pressure-levels",
                request,
                str(chunk_file),
            )

            size_mb = chunk_file.stat().st_size / 1e6
            print(f"[CDS] Downloaded {chunk_file.name} ({size_mb:.1f} MB)")

    # Merge all chunks into final JJA file (serialised to avoid OOM)
    print(f"[CDS] Waiting for merge lock -> {out_file.name}")
    with _MERGE_LOCK:
        print(f"[CDS] Merging {len(chunk_files)} chunks -> {out_file.name}")
        _merge_chunks(chunk_files, out_file)

    # Clean up monthly chunk files
    for f in chunk_files:
        f.unlink(missing_ok=True)
    print(f"[CDS] Cleaned up {len(chunk_files)} chunk files")

    size_mb = out_file.stat().st_size / 1e6
    print(f"[CDS] Final: {out_file.name} ({size_mb:.1f} MB)")
    return out_file


def download_humidity_only(
    region: str,
    year: int,
    months: list[int],
    out_dir: Path,
) -> Path:
    """Download only specific_humidity and merge into an existing T+Z NetCDF.

    Skips if the existing file already contains ``specific_humidity``.
    Returns the path to the (augmented) NetCDF file.
    """
    import cdsapi
    import xarray as xr

    region = validate_region(region)
    month_tag = "_".join(f"{m:02d}" for m in sorted(months))
    existing_file = out_dir / f"era5_pressure_{region}_{year}_{month_tag}.nc"

    if not existing_file.exists():
        raise FileNotFoundError(f"No existing file to augment: {existing_file}")

    # Check if already has humidity
    with xr.open_dataset(existing_file) as ds:
        if "q" in ds.data_vars or "specific_humidity" in ds.data_vars:
            print(f"[skip] {existing_file.name} already has specific_humidity")
            return existing_file

    bbox = REGION_BBOX.get(region)
    if bbox is None:
        raise ValueError(f"No bounding box for region {region!r}")

    client = cdsapi.Client()
    q_chunks: list[Path] = []

    for month in sorted(months):
        for ci, days in enumerate(_day_chunks(year, month, _CHUNK_DAYS_Q)):
            q_file = out_dir / f"era5_q_only_{region}_{year}_{month:02d}_chunk{ci}.nc"
            q_chunks.append(q_file)

            if q_file.exists():
                print(f"[skip] {q_file.name} already exists")
                continue

            request = {
                "product_type": "reanalysis",
                "variable": ["specific_humidity"],
                "pressure_level": [str(p) for p in PRESSURE_LEVELS_HPA],
                "year": str(year),
                "month": f"{month:02d}",
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": bbox,
                "data_format": "netcdf",
            }

            n_fields = 1 * len(PRESSURE_LEVELS_HPA) * len(days) * 24
            print(f"[CDS-q] {region}/{year}/{month:02d} chunk{ci}: {n_fields} fields")

            client.retrieve(
                "reanalysis-era5-pressure-levels",
                request,
                str(q_file),
            )
            size_mb = q_file.stat().st_size / 1e6
            print(f"[CDS-q] Downloaded {q_file.name} ({size_mb:.1f} MB)")

    # Merge humidity chunks into one dataset
    q_datasets = [_normalize_cds(xr.open_dataset(f)) for f in q_chunks]
    q_merged = xr.concat(q_datasets, dim="time").sortby("time")

    # Merge with existing T+Z file
    print(f"[CDS-q] Merging humidity into {existing_file.name}")
    with _MERGE_LOCK:
        existing_ds = _normalize_cds(xr.open_dataset(existing_file))
        merged = xr.merge([existing_ds, q_merged])
        # Write to temp then atomic rename
        tmp_file = existing_file.with_suffix(".tmp.nc")
        encoding = {v: {"_FillValue": None} for v in merged.data_vars}
        merged.to_netcdf(tmp_file, encoding=encoding)
        existing_ds.close()
        for ds in q_datasets:
            ds.close()
        q_merged.close()
        merged.close()
        tmp_file.rename(existing_file)

    # Clean up q-only chunks
    for f in q_chunks:
        f.unlink(missing_ok=True)

    size_mb = existing_file.stat().st_size / 1e6
    print(f"[CDS-q] Augmented: {existing_file.name} ({size_mb:.1f} MB)")
    return existing_file


def _normalize_cds(ds):
    """Normalize new CDS netCDF4 format (valid_time) to legacy conventions.

    CDS files come in three flavours:
    1. Legacy: only ``time`` dim -> nothing to do.
    2. New CDS: only ``valid_time`` dim -> rename to ``time``.
    3. Hybrid (chunked concat artifact): both ``time`` (small batch dim)
       **and** ``valid_time`` (actual timestamps).  Collapse the artifact
       ``time`` dim via nanmean, then rename ``valid_time`` -> ``time``.
    """
    if "valid_time" in ds.dims and "time" in ds.dims:
        # Case 3 - hybrid.  ``time`` is a batch artifact; collapse it.
        ds = ds.mean(dim="time")          # collapses artifact dim
        ds = ds.rename_dims({"valid_time": "time"})
        if "valid_time" in ds.coords:
            ds = ds.rename_vars({"valid_time": "time"})
    elif "valid_time" in ds.dims:
        # Case 2 - pure new CDS format.
        ds = ds.rename_dims({"valid_time": "time"})
        if "valid_time" in ds.coords:
            ds = ds.rename_vars({"valid_time": "time"})
    ds = ds.drop_vars(["expver", "number"], errors="ignore")
    return ds


def _merge_chunks(chunk_files: list[Path], out_file: Path) -> None:
    """Concatenate NetCDF chunks along the time dimension."""
    import xarray as xr

    datasets = [_normalize_cds(xr.open_dataset(f)) for f in chunk_files]
    merged = xr.concat(datasets, dim="time")
    merged = merged.sortby("time")
    # Explicit encoding avoids memory doubling from fillna (xarray #7397)
    encoding = {v: {"_FillValue": None} for v in merged.data_vars}
    merged.to_netcdf(out_file, encoding=encoding)
    for ds in datasets:
        ds.close()
    merged.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ERA5 pressure-level data (T + Z + q) via CDS API",
    )
    add_region_arg(parser)
    parser.add_argument("--year", type=int, required=True, help="Year to download")
    parser.add_argument(
        "--months",
        type=str,
        default="06,07,08",
        help="Comma-separated months (default: 06,07,08)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for NetCDF files",
    )
    parser.add_argument(
        "--humidity-only",
        action="store_true",
        help="Only download specific_humidity and merge into existing T+Z file",
    )
    args = parser.parse_args()

    months = parse_months(args.months)
    if args.humidity_only:
        download_humidity_only(
            region=args.region,
            year=args.year,
            months=months,
            out_dir=args.out_dir,
        )
    else:
        download_pressure_levels(
            region=args.region,
            year=args.year,
            months=months,
            out_dir=args.out_dir,
        )


if __name__ == "__main__":
    main()
