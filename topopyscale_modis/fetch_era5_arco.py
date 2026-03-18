"""Download ERA5 pressure-level data from ARCO-ERA5 (Google Cloud Zarr).

ARCO-ERA5 provides the same ERA5 reanalysis data as CDS but served as
cloud-native Zarr on Google Cloud Storage -- no queue, no tape archive,
instant access.

Reference: https://cloud.google.com/storage/docs/public-datasets/era5

Usage:
    from topopyscale_modis.fetch_era5_arco import download_from_arco
    download_from_arco(region="02-03", year=2023, months=[6,7,8], out_dir=Path("/db/era5"))
"""

from __future__ import annotations

from pathlib import Path

from topopyscale_modis.config import (
    CDS_VARIABLES,
    PRESSURE_LEVELS_HPA,
    REGION_BBOX,
    validate_region,
)

ARCO_ZARR_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"


def download_from_arco(
    region: str,
    year: int,
    months: list[int],
    out_dir: Path,
) -> Path:
    """Download ERA5 pressure-level T + geopotential from ARCO-ERA5 Zarr.

    Produces a NetCDF file identical in structure to the CDS download:
    dims [time, level, latitude, longitude], variables temperature + geopotential,
    longitude in -180/+180 convention.

    Returns the path to the output NetCDF file.
    """
    import numpy as np
    import xarray as xr

    region = validate_region(region)
    out_dir.mkdir(parents=True, exist_ok=True)

    month_tag = "_".join(f"{m:02d}" for m in sorted(months))
    out_file = out_dir / f"era5_pressure_{region}_{year}_{month_tag}.nc"

    if out_file.exists():
        print(f"[arco] [skip] {out_file.name} already exists")
        return out_file

    bbox = REGION_BBOX.get(region)
    if bbox is None:
        raise ValueError(
            f"No bounding box for region {region!r}. "
            f"Add it to topopyscale_modis.config.REGION_BBOX"
        )

    north, west, south, east = bbox

    # ARCO-ERA5 uses 0-360 longitude; convert from -180/+180
    west_360 = west % 360
    east_360 = east % 360

    # Open ARCO-ERA5 Zarr store (anonymous access, NO Dask)
    # chunks=None avoids building a Dask task graph (~1s vs 12+ min)
    # storage_options lets fsspec handle GCS internally
    print(f"[arco] Opening ARCO-ERA5 Zarr for {region}/{year} months={sorted(months)}")
    ds = xr.open_zarr(
        ARCO_ZARR_URL,
        chunks=None,
        storage_options=dict(token="anon"),
        consolidated=True,
    )

    # Select only needed variables + pressure levels immediately
    ds = ds[CDS_VARIABLES]
    ds = ds.sel(level=PRESSURE_LEVELS_HPA)

    # Select spatial subset
    # ARCO latitude is descending (90 to -90)
    lat_sel = slice(north, south)  # descending order
    if west_360 <= east_360:
        lon_sel = slice(west_360, east_360)
        ds = ds.sel(latitude=lat_sel, longitude=lon_sel)
    else:
        # Crosses the antimeridian (e.g., Alaska 01-03: west=183, east=209)
        lon_sel = slice(west_360, east_360)
        ds = ds.sel(latitude=lat_sel, longitude=lon_sel)

    # Load month by month to control memory
    # NOTE: ARCO chunks are global snapshots (721x1440 per timestep),
    # so each .load() downloads the full grid then subsets -- this is
    # an inherent limitation of the store layout.
    loaded_months = []
    for m in sorted(months):
        time_sel = slice(f"{year}-{m:02d}-01",
                         f"{year}-{m:02d}-28" if m == 2
                         else f"{year}-{m:02d}-30" if m in (4, 6, 9, 11)
                         else f"{year}-{m:02d}-31")
        ds_month = ds.sel(time=time_sel)
        print(f"[arco] Loading {year}-{m:02d} ({ds_month.sizes}) ...")
        loaded_months.append(ds_month.load())

    ds_loaded = xr.concat(loaded_months, dim="time")

    # Remove any duplicate times
    _, unique_idx = np.unique(ds_loaded.time.values, return_index=True)
    ds_loaded = ds_loaded.isel(time=np.sort(unique_idx))

    # Convert longitude from 0-360 to -180/+180 for toposcale.py compatibility
    lon_vals = ds_loaded.longitude.values
    lon_converted = np.where(lon_vals > 180, lon_vals - 360, lon_vals)
    ds_loaded = ds_loaded.assign_coords(longitude=lon_converted)
    # Sort longitude so it's monotonically increasing
    ds_loaded = ds_loaded.sortby("longitude")

    # Write to NetCDF
    encoding = {v: {"_FillValue": None} for v in ds_loaded.data_vars}
    print(f"[arco] Writing {out_file.name} ...")
    ds_loaded.to_netcdf(out_file, encoding=encoding)
    ds_loaded.close()

    size_mb = out_file.stat().st_size / 1e6
    print(f"[arco] Done: {out_file.name} ({size_mb:.1f} MB)")
    return out_file
