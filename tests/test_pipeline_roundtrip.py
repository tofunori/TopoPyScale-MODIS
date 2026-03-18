"""End-to-end roundtrip test for the full pipeline."""

import duckdb
import pandas as pd

from topopyscale_modis.ingest import ingest_era5
from topopyscale_modis.partition import partition_rain_snow
from topopyscale_modis.toposcale import interpolate_temperature, load_pixel_elevations


class TestFullPipeline:
    def test_roundtrip(self, tmp_path, test_db, era5_nc, precip_df):
        """Run the entire pipeline: toposcale -> partition -> ingest without errors."""
        # Step 1: Load pixels
        pixels = load_pixel_elevations(test_db, "02-03")
        assert len(pixels) == 10

        # Step 2: TopoSCALE interpolation
        temp_df = interpolate_temperature(era5_nc, pixels)
        assert len(temp_df) == 10 * 48  # 10 pixels x 48 hours

        # Step 3: Rain/snow partition
        part_df = partition_rain_snow(temp_df, precip_df)
        assert len(part_df) == 10 * 2 * 5  # 10 pixels x 2 days x 5 thresholds

        # Step 4: Save to CSV and ingest
        temp_csv = tmp_path / "temperature.csv"
        part_csv = tmp_path / "partition.csv"
        temp_df.to_csv(temp_csv, index=False)
        part_df.to_csv(part_csv, index=False)

        ingest_era5(test_db, temperature_csv=temp_csv, partition_csv=part_csv)

        # Step 5: Verify final state
        con = duckdb.connect(str(test_db), read_only=True)

        n_temp = con.execute("SELECT count(*) FROM era5.temperature_at_pixel").fetchone()[0]
        assert n_temp == 480  # 10 x 48

        n_part = con.execute("SELECT count(*) FROM era5.snow_partition_corrected").fetchone()[0]
        assert n_part == 100  # 10 x 2 x 5

        # The view should be queryable (returns 0 rows since obs is empty, but no errors)
        result = con.execute("SELECT * FROM modis.obs_with_era5 LIMIT 1").fetchdf()
        assert isinstance(result, pd.DataFrame)

        con.close()
