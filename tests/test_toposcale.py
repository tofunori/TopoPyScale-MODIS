"""Tests for topopyscale_modis.toposcale."""

from topopyscale_modis.toposcale import interpolate_temperature, load_pixel_elevations


class TestToposcale:
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
