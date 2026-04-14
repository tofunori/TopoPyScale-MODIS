"""Tests for topopyscale_modis.toposcale."""

import numpy as np
import pandas as pd
import pytest

from topopyscale_modis.toposcale import (
    _patch_weights_geographic,
    interpolate_temperature,
    load_pixel_elevations,
)


class TestToposcale:
    def test_patch_weights_account_for_longitude_convergence(self):
        longitudes = np.array([-100.5, -100.0, -99.5])

        _, _, weights_equator = _patch_weights_geographic(
            np.array([0.5, 0.0, -0.5]),
            longitudes,
            station_lat=0.0,
            station_lon=-100.25,
        )
        _, _, weights_high_lat = _patch_weights_geographic(
            np.array([60.5, 60.0, 59.5]),
            longitudes,
            station_lat=60.0,
            station_lon=-100.25,
        )

        # At 60 N, the same longitude offset spans fewer metres than at the
        # equator, so the same-latitude western neighbor should gain weight.
        assert weights_high_lat[1, 1] > weights_equator[1, 1]
        assert np.isclose(weights_high_lat.sum(), 1.0)

    def test_interpolation_temperature_range(self, era5_nc, test_db):
        pixels = load_pixel_elevations(test_db, "02-03")
        assert len(pixels) == 10

        result = interpolate_temperature(era5_nc, pixels)
        assert not result.empty
        assert set(result.columns) >= {
            "pixel_id", "date", "hour", "T_pixel_K", "T_pixel_C", "z_pixel_m", "lapse_rate_Ckm",
        }

        # Temperatures should be between -20 C and +25 C for 1500-3500 m elevations
        assert result["T_pixel_C"].min() > -20.0, "Temperature below -20 C is unrealistic"
        assert result["T_pixel_C"].max() < 25.0, "Temperature above 25 C is unrealistic"

    def test_lapse_rate_variable(self, era5_nc, test_db):
        pixels = load_pixel_elevations(test_db, "02-03")
        result = interpolate_temperature(era5_nc, pixels)

        lr = result["lapse_rate_Ckm"].dropna()
        assert len(lr) > 0, "Expected some rows with lapse rate"

        lr_abs = lr.abs()
        assert lr_abs.min() >= 2.0, f"Lapse rate too small: {lr_abs.min()}"
        assert lr_abs.max() <= 12.0, f"Lapse rate too large: {lr_abs.max()}"

        # Should NOT be a fixed constant -- diurnal variation causes slight changes
        assert lr.nunique() > 1, "Lapse rate should vary across timesteps"

    def test_all_pixels_have_results(self, era5_nc, test_db):
        pixels = load_pixel_elevations(test_db, "02-03")
        result = interpolate_temperature(era5_nc, pixels)

        assert result["pixel_id"].nunique() == 10

        counts = result.groupby("pixel_id").size()
        assert (counts == 48).all(), f"Expected 48 records per pixel, got: {counts.unique()}"

    def test_refuses_silent_extrapolation_above_profile(self, era5_nc):
        pixels = pd.DataFrame(
            [
                {
                    "pixel_id": 999,
                    "elevation": 22000.0,
                    "latitude": 52.0,
                    "longitude": -118.0,
                }
            ]
        )

        with pytest.raises(ValueError, match="Refusing silent extrapolation above the pressure profile"):
            interpolate_temperature(era5_nc, pixels)
