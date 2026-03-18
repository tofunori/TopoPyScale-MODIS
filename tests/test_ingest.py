"""Tests for topopyscale_modis.ingest."""

import duckdb

from topopyscale_modis.ingest import ingest_era5
from topopyscale_modis.partition import partition_rain_snow
from topopyscale_modis.toposcale import interpolate_temperature, load_pixel_elevations


class TestIngest:
    def test_creates_era5_tables(self, tmp_path, test_db, era5_nc, precip_df):
        pixels = load_pixel_elevations(test_db, "02-03")
        temp_df = interpolate_temperature(era5_nc, pixels)
        part_df = partition_rain_snow(temp_df, precip_df)

        temp_csv = tmp_path / "temperature.csv"
        part_csv = tmp_path / "partition.csv"
        temp_df.to_csv(temp_csv, index=False)
        part_df.to_csv(part_csv, index=False)

        ingest_era5(test_db, temperature_csv=temp_csv, partition_csv=part_csv)

        con = duckdb.connect(str(test_db), read_only=True)
        tables = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'era5'"
            ).fetchall()
        }
        assert "temperature_at_pixel" in tables
        assert "snow_partition_corrected" in tables

        n_temp = con.execute("SELECT count(*) FROM era5.temperature_at_pixel").fetchone()[0]
        n_part = con.execute("SELECT count(*) FROM era5.snow_partition_corrected").fetchone()[0]
        assert n_temp > 0
        assert n_part > 0
        con.close()

    def test_obs_with_era5_view_exists(self, tmp_path, test_db, era5_nc, precip_df):
        pixels = load_pixel_elevations(test_db, "02-03")
        temp_df = interpolate_temperature(era5_nc, pixels)
        part_df = partition_rain_snow(temp_df, precip_df)

        temp_csv = tmp_path / "temperature.csv"
        part_csv = tmp_path / "partition.csv"
        temp_df.to_csv(temp_csv, index=False)
        part_df.to_csv(part_csv, index=False)

        ingest_era5(test_db, temperature_csv=temp_csv, partition_csv=part_csv)

        con = duckdb.connect(str(test_db), read_only=True)
        views = {
            row[0]
            for row in con.execute(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'modis' AND table_type = 'VIEW'"
            ).fetchall()
        }
        assert "obs_with_era5" in views
        con.close()
