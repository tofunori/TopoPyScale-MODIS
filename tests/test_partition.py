"""Tests for topopyscale_modis.partition."""

import pytest

from topopyscale_modis.config import DEFAULT_THRESHOLDS_C
from topopyscale_modis.partition import partition_rain_snow
from topopyscale_modis.toposcale import interpolate_temperature, load_pixel_elevations


class TestPartition:
    def test_multi_threshold_output(self, era5_nc, test_db, precip_df):
        pixels = load_pixel_elevations(test_db, "02-03")
        temp_df = interpolate_temperature(era5_nc, pixels)
        result = partition_rain_snow(temp_df, precip_df)

        thresholds_found = sorted(result["threshold_C"].unique())
        assert thresholds_found == DEFAULT_THRESHOLDS_C

        # 10 pixels x 2 days x 5 thresholds = 100 rows
        assert len(result) == 10 * 2 * 5

    def test_snow_increases_with_higher_threshold(self, era5_nc, test_db, precip_df):
        """Higher threshold means more hours classified as snow -> more total snow."""
        pixels = load_pixel_elevations(test_db, "02-03")
        temp_df = interpolate_temperature(era5_nc, pixels)
        result = partition_rain_snow(temp_df, precip_df)

        snow_by_thresh = result.groupby("threshold_C")["snow_corrected_m"].sum()
        for i in range(len(DEFAULT_THRESHOLDS_C) - 1):
            lo = DEFAULT_THRESHOLDS_C[i]
            hi = DEFAULT_THRESHOLDS_C[i + 1]
            assert snow_by_thresh[hi] >= snow_by_thresh[lo], (
                f"Snow at threshold {hi} C should be >= snow at {lo} C"
            )

    def test_rain_plus_snow_equals_total(self, era5_nc, test_db, precip_df):
        """For each (pixel, date, threshold), rain + snow = total precip."""
        pixels = load_pixel_elevations(test_db, "02-03")
        temp_df = interpolate_temperature(era5_nc, pixels)
        result = partition_rain_snow(temp_df, precip_df)

        daily_total = precip_df.groupby(["pixel_id", "date"])["total_precip_m"].sum().reset_index()

        for _, row in result.iterrows():
            expected = daily_total.loc[
                (daily_total["pixel_id"] == row["pixel_id"])
                & (daily_total["date"] == row["date"]),
                "total_precip_m",
            ].values[0]
            actual = row["snow_corrected_m"] + row["rain_corrected_m"]
            assert actual == pytest.approx(expected, abs=1e-9), (
                f"rain+snow != total for pixel {row['pixel_id']}, date {row['date']}, "
                f"threshold {row['threshold_C']}"
            )
