"""Shared fixtures for TopoPyScale-MODIS tests.

Provides synthetic ERA5 NetCDF, DuckDB, and precipitation data.
"""

from __future__ import annotations

import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from topopyscale_modis.config import PRESSURE_LEVELS_HPA, G

_LEVELS = np.array(PRESSURE_LEVELS_HPA, dtype=float)


def _altitude_from_pressure(p_hpa: float) -> float:
    """Barometric formula (simplified): altitude in metres from pressure in hPa."""
    return -8434.5 * math.log(p_hpa / 1013.25)


_ALT_BY_LEVEL = np.array([_altitude_from_pressure(p) for p in PRESSURE_LEVELS_HPA])


def _make_geopotential(altitudes: np.ndarray) -> np.ndarray:
    """Convert altitude (m) to geopotential (m^2/s^2)."""
    return altitudes * G


@pytest.fixture()
def era5_nc(tmp_path: Path) -> Path:
    """Create a small synthetic ERA5 pressure-level NetCDF.

    Dimensions: time=48 (2 days x 24 h), level=16, lat=3, lon=3.
    Region 02-03 bbox: N59, W-129, S48, E-114 -- we place grid inside.
    """
    hours_per_day = 24
    n_days = 2
    n_times = hours_per_day * n_days

    times = pd.date_range("2023-07-01", periods=n_times, freq="h")
    levels = _LEVELS
    lats = np.array([50.0, 52.0, 54.0])
    lons = np.array([-120.0, -118.0, -116.0])

    shape = (n_times, len(levels), len(lats), len(lons))

    alt_4d = np.broadcast_to(_ALT_BY_LEVEL[np.newaxis, :, np.newaxis, np.newaxis], shape)
    geopotential = _make_geopotential(alt_4d)

    t_surface = 288.0  # K
    lapse_rate = 0.0065  # K/m
    temperature = np.empty(shape, dtype=np.float64)
    for ti in range(n_times):
        hour = times[ti].hour
        diurnal_base = 3.0 * np.sin(2 * np.pi * (hour - 6) / 24)
        for li in range(len(levels)):
            alt_km = _ALT_BY_LEVEL[li] / 1000.0
            diurnal = diurnal_base * max(0.0, 1.0 - 0.15 * alt_km)
            temperature[ti, li, :, :] = t_surface - lapse_rate * _ALT_BY_LEVEL[li] + diurnal

    ds = xr.Dataset(
        {
            "t": (["time", "level", "latitude", "longitude"], temperature),
            "z": (["time", "level", "latitude", "longitude"], geopotential),
        },
        coords={
            "time": times,
            "level": levels,
            "latitude": lats,
            "longitude": lons,
        },
    )

    nc_path = tmp_path / "era5_synth.nc"
    ds.to_netcdf(nc_path)
    ds.close()
    return nc_path


@pytest.fixture()
def test_db(tmp_path: Path) -> Path:
    """Create a mini DuckDB with modis.pixel_static (10 pixels) and modis.obs (empty)."""
    db_path = tmp_path / "test.duckdb"
    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS modis")

    con.execute(
        """
        CREATE TABLE modis.pixel_static (
            pixel_id   BIGINT,
            region     VARCHAR,
            run_tag    VARCHAR,
            latitude   DOUBLE,
            longitude  DOUBLE,
            elevation  DOUBLE,
            glacier_fraction DOUBLE,
            slope      DOUBLE,
            aspect     DOUBLE
        )
        """
    )
    for pid in range(1, 11):
        elev = 1500 + (pid - 1) * 222
        con.execute(
            "INSERT INTO modis.pixel_static VALUES (?, '02_03', 'all', 52.0, -118.0, ?, 0.8, 10.0, 180.0)",
            [pid, elev],
        )

    con.execute(
        """
        CREATE TABLE modis.obs (
            pixel_id  BIGINT,
            region    VARCHAR,
            run_tag   VARCHAR,
            date      DATE,
            method    VARCHAR,
            qa_mode   VARCHAR,
            albedo    DOUBLE
        )
        """
    )

    con.close()
    return db_path


@pytest.fixture()
def precip_df() -> pd.DataFrame:
    """Hourly precipitation DataFrame matching the NetCDF time span.

    10 pixels x 48 hours, total_precip_m in [0, 0.005] m/h.
    """
    rng = np.random.default_rng(42)
    rows = []
    for pid in range(1, 11):
        for day_offset in range(2):
            date_str = f"2023-07-0{day_offset + 1}"
            for hour in range(24):
                rows.append(
                    {
                        "pixel_id": pid,
                        "date": date_str,
                        "hour": hour,
                        "total_precip_m": round(rng.uniform(0, 0.005), 6),
                    }
                )
    return pd.DataFrame(rows)
