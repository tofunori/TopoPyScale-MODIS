"""Ingest ERA5 results into DuckDB (schema era5) and create cross-schema views.

Creates:
  - era5.temperature_at_pixel  (TopoSCALE daily T + lapse rate)
  - era5.precipitation_era5land (raw ERA5-Land precip + snowfall)
  - era5.snow_partition_corrected (re-partitioned snow by threshold)
  - modis.obs_with_era5  (JOIN albedo + T + precip + corrected snow)

Usage:
    era5-ingest --db modis.duckdb --temperature-csv temp.csv --partition-csv part.csv
    python -m topopyscale_modis.ingest ...
    from topopyscale_modis.ingest import ingest_era5
"""

from __future__ import annotations

import argparse
from pathlib import Path

from topopyscale_modis._cli import add_db_arg


def _sql_str(value: str) -> str:
    return value.replace("'", "''")


def ingest_era5(
    db_path: Path,
    temperature_csv: Path | None = None,
    era5land_csv: Path | None = None,
    partition_csv: Path | None = None,
    dewpoint_csv: Path | None = None,
) -> None:
    """Ingest ERA5 CSV outputs into DuckDB.

    Each CSV argument is optional -- only the provided ones are ingested.
    The cross-schema view modis.obs_with_era5 is always (re)created.
    """
    import duckdb

    con = duckdb.connect(str(db_path))
    con.execute("CREATE SCHEMA IF NOT EXISTS era5")

    if temperature_csv is not None:
        print(f"[ingest] Loading temperature: {temperature_csv}")
        csv_path = _sql_str(str(temperature_csv))
        # Create table if it doesn't exist, otherwise append
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS era5.temperature_at_pixel (
                pixel_id BIGINT,
                date DATE,
                hour INTEGER,
                t_pixel_k DOUBLE,
                t_pixel_c DOUBLE,
                z_pixel_m DOUBLE,
                lapse_rate_c_per_km DOUBLE,
                q_pixel_kgkg DOUBLE,
                p_pixel_hpa DOUBLE,
                td_pixel_c DOUBLE,
                rh_pixel_pct DOUBLE
            )
            """
        )
        # Add humidity columns to existing tables that don't have them yet
        for col, typ in [
            ("q_pixel_kgkg", "DOUBLE"),
            ("p_pixel_hpa", "DOUBLE"),
            ("td_pixel_c", "DOUBLE"),
            ("rh_pixel_pct", "DOUBLE"),
        ]:
            try:
                con.execute(f"ALTER TABLE era5.temperature_at_pixel ADD COLUMN {col} {typ}")
            except Exception:
                pass  # column already exists
        # Detect which columns exist in the CSV
        result = con.execute(f"SELECT * FROM read_csv_auto('{csv_path}') LIMIT 0")
        csv_cols = {c[0].lower() for c in result.description}
        has_humidity = "q_pixel_kgkg" in csv_cols
        if has_humidity:
            con.execute(
                f"""
                INSERT INTO era5.temperature_at_pixel
                SELECT
                    pixel_id::BIGINT,
                    date::DATE,
                    hour::INTEGER,
                    T_pixel_K::DOUBLE,
                    T_pixel_C::DOUBLE,
                    z_pixel_m::DOUBLE,
                    lapse_rate_Ckm::DOUBLE,
                    q_pixel_kgkg::DOUBLE,
                    P_pixel_hPa::DOUBLE,
                    Td_pixel_C::DOUBLE,
                    RH_pixel_pct::DOUBLE
                FROM read_csv_auto('{csv_path}')
                """
            )
        else:
            con.execute(
                f"""
                INSERT INTO era5.temperature_at_pixel
                    (pixel_id, date, hour, t_pixel_k, t_pixel_c, z_pixel_m, lapse_rate_c_per_km)
                SELECT
                    pixel_id::BIGINT,
                    date::DATE,
                    hour::INTEGER,
                    T_pixel_K::DOUBLE,
                    T_pixel_C::DOUBLE,
                    z_pixel_m::DOUBLE,
                    lapse_rate_Ckm::DOUBLE
                FROM read_csv_auto('{csv_path}')
                """
            )
        n = con.execute("SELECT count(*) FROM era5.temperature_at_pixel").fetchone()[0]
        print(f"[ingest] era5.temperature_at_pixel: {n} rows (humidity={'yes' if has_humidity else 'no'})")

    if era5land_csv is not None:
        print(f"[ingest] Loading ERA5-Land precip: {era5land_csv}")
        csv_path = _sql_str(str(era5land_csv))
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS era5.precipitation_era5land (
                pixel_id BIGINT,
                date DATE,
                total_precip_m DOUBLE,
                snowfall_m DOUBLE,
                temperature_2m_k DOUBLE
            )
            """
        )
        con.execute(
            f"""
            INSERT INTO era5.precipitation_era5land
            SELECT
                pixel_id::BIGINT,
                date::DATE,
                total_precip_m::DOUBLE,
                snowfall_m::DOUBLE,
                temperature_2m_K::DOUBLE
            FROM read_csv_auto('{csv_path}', union_by_name=true)
            """
        )
        n = con.execute("SELECT count(*) FROM era5.precipitation_era5land").fetchone()[0]
        print(f"[ingest] era5.precipitation_era5land: {n} rows")

    if dewpoint_csv is not None:
        print(f"[ingest] Loading dewpoint: {dewpoint_csv}")
        csv_path = _sql_str(str(dewpoint_csv))
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS era5.dewpoint_era5land (
                pixel_id BIGINT,
                date DATE,
                dewpoint_2m_k DOUBLE
            )
            """
        )
        con.execute(
            f"""
            INSERT INTO era5.dewpoint_era5land
            SELECT
                pixel_id::BIGINT,
                date::DATE,
                dewpoint_2m_K::DOUBLE
            FROM read_csv_auto('{csv_path}')
            """
        )
        n = con.execute("SELECT count(*) FROM era5.dewpoint_era5land").fetchone()[0]
        print(f"[ingest] era5.dewpoint_era5land: {n} rows")

    if partition_csv is not None:
        print(f"[ingest] Loading partition: {partition_csv}")
        csv_path = _sql_str(str(partition_csv))
        # Detect if CSV has rh_pct column (Jennings adaptive) or not (fixed thresholds)
        sample_result = con.execute(f"SELECT * FROM read_csv_auto('{csv_path}') LIMIT 1")
        has_rh = "rh_pct" in [c[0].lower() for c in sample_result.description]
        con.execute("DROP TABLE IF EXISTS era5.snow_partition_corrected")
        if has_rh:
            con.execute(
                f"""
                CREATE TABLE era5.snow_partition_corrected AS
                SELECT
                    pixel_id::BIGINT AS pixel_id,
                    date::DATE AS date,
                    threshold_C::DOUBLE AS threshold_c,
                    rh_pct::DOUBLE AS rh_pct,
                    snow_corrected_m::DOUBLE AS snow_corrected_m,
                    rain_corrected_m::DOUBLE AS rain_corrected_m
                FROM read_csv_auto('{csv_path}')
                """
            )
            print("[ingest] Jennings adaptive partition detected (rh_pct column)")
        else:
            con.execute(
                f"""
                CREATE TABLE era5.snow_partition_corrected AS
                SELECT
                    pixel_id::BIGINT AS pixel_id,
                    date::DATE AS date,
                    threshold_C::DOUBLE AS threshold_c,
                    snow_corrected_m::DOUBLE AS snow_corrected_m,
                    rain_corrected_m::DOUBLE AS rain_corrected_m
                FROM read_csv_auto('{csv_path}')
                """
            )
        n = con.execute("SELECT count(*) FROM era5.snow_partition_corrected").fetchone()[0]
        print(f"[ingest] era5.snow_partition_corrected: {n} rows")

    # Cross-schema view: modis.obs + era5 temperature + partition (threshold=1.5 default)
    _create_obs_with_era5_view(con)

    con.close()
    print(f"[ingest] Done: {db_path}")


def _create_obs_with_era5_view(con) -> None:
    """Create modis.obs_with_era5 view joining albedo + ERA5 data."""
    # Check which ERA5 tables exist
    tables = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'era5'"
        ).fetchall()
    }

    if not tables:
        print("[ingest] No ERA5 tables found -- skipping obs_with_era5 view")
        return

    # Build the view with available tables
    # Use daily average T for the join (average across hours)
    joins = []
    extra_cols = []

    if "temperature_at_pixel" in tables:
        # Check if humidity columns exist in the table
        temp_cols = {
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'era5' AND table_name = 'temperature_at_pixel'"
            ).fetchall()
        }
        humidity_aggs = ""
        humidity_cols = []
        if "rh_pixel_pct" in temp_cols:
            humidity_aggs = (
                ",\n                       avg(td_pixel_c) AS td_pixel_c_daily"
                ",\n                       avg(rh_pixel_pct) AS rh_pixel_pct_daily"
                ",\n                       any_value(p_pixel_hpa) AS p_pixel_hpa"
                ",\n                       any_value(q_pixel_kgkg) AS q_pixel_kgkg"
            )
            humidity_cols = [
                "t.td_pixel_c_daily", "t.rh_pixel_pct_daily",
                "t.p_pixel_hpa", "t.q_pixel_kgkg",
            ]
        joins.append(
            f"""
            LEFT JOIN (
                SELECT pixel_id, date,
                       avg(t_pixel_c) AS t_pixel_c_daily,
                       avg(lapse_rate_c_per_km) AS lapse_rate_c_per_km,
                       any_value(z_pixel_m) AS z_pixel_m{humidity_aggs}
                FROM era5.temperature_at_pixel
                GROUP BY pixel_id, date
            ) t ON o.pixel_id = t.pixel_id AND o.date = t.date
            """
        )
        extra_cols.extend(["t.t_pixel_c_daily", "t.lapse_rate_c_per_km", "t.z_pixel_m"])
        extra_cols.extend(humidity_cols)

    if "precipitation_era5land" in tables:
        joins.append(
            "LEFT JOIN era5.precipitation_era5land pl ON o.pixel_id = pl.pixel_id AND o.date = pl.date"
        )
        extra_cols.extend(["pl.total_precip_m", "pl.snowfall_m", "pl.temperature_2m_k"])

    if "dewpoint_era5land" in tables:
        joins.append(
            "LEFT JOIN era5.dewpoint_era5land dw ON o.pixel_id = dw.pixel_id AND o.date = dw.date"
        )
        extra_cols.append("dw.dewpoint_2m_k")
        # Compute RH(%) from T2m and dewpoint using Magnus formula:
        # RH = 100 * exp(17.625 * Td_C / (243.04 + Td_C)) / exp(17.625 * T_C / (243.04 + T_C))
        # Use ERA5-Land T2m (from precip table) if available, else use TopoSCALE T
        if "precipitation_era5land" in tables:
            extra_cols.append(
                "100.0 * exp(17.625 * (dw.dewpoint_2m_k - 273.15) / (243.04 + dw.dewpoint_2m_k - 273.15))"
                " / exp(17.625 * (pl.temperature_2m_k - 273.15) / (243.04 + pl.temperature_2m_k - 273.15))"
                " AS rh_pct"
            )
        elif "temperature_at_pixel" in tables:
            extra_cols.append(
                "100.0 * exp(17.625 * (dw.dewpoint_2m_k - 273.15) / (243.04 + dw.dewpoint_2m_k - 273.15))"
                " / exp(17.625 * t.t_pixel_c_daily / (243.04 + t.t_pixel_c_daily))"
                " AS rh_pct"
            )

    if "snow_partition_corrected" in tables:
        # Check if Jennings adaptive (has rh_pct -> 1 row per pixel x day)
        # or fixed multi-threshold (filter on threshold_c = 1.5)
        sp_cols = {
            row[0]
            for row in con.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = 'era5' AND table_name = 'snow_partition_corrected'"
            ).fetchall()
        }
        if "rh_pct" in sp_cols:
            # Jennings adaptive: 1 row per pixel x day, no threshold filter needed
            joins.append(
                "LEFT JOIN era5.snow_partition_corrected sp"
                " ON o.pixel_id = sp.pixel_id AND o.date = sp.date"
            )
            extra_cols.extend([
                "sp.threshold_c AS jennings_threshold_c",
                "sp.rh_pct AS jennings_rh_pct",
                "sp.snow_corrected_m", "sp.rain_corrected_m",
            ])
        else:
            # Fixed multi-threshold: default to 1.5C
            joins.append(
                """
                LEFT JOIN era5.snow_partition_corrected sp
                  ON o.pixel_id = sp.pixel_id AND o.date = sp.date AND sp.threshold_c = 1.5
                """
            )
            extra_cols.extend(["sp.snow_corrected_m", "sp.rain_corrected_m"])

    if "snow_partition_jennings_logistic" in tables:
        joins.append(
            "LEFT JOIN era5.snow_partition_jennings_logistic jl"
            " ON o.pixel_id = jl.pixel_id AND o.date = jl.date"
        )
        extra_cols.extend([
            "jl.p_snow_mean AS jennings_logistic_p_snow",
            "jl.snow_corrected_m AS jennings_logistic_snow_m",
            "jl.rain_corrected_m AS jennings_logistic_rain_m",
        ])

    if not extra_cols:
        return

    extra_select = ",\n          ".join(extra_cols)
    join_sql = "\n        ".join(joins)

    con.execute("CREATE SCHEMA IF NOT EXISTS modis")
    con.execute(
        f"""
        CREATE OR REPLACE VIEW modis.obs_with_era5 AS
        SELECT
          o.*,
          {extra_select}
        FROM modis.obs o
        {join_sql}
        """
    )
    print("[ingest] Created view modis.obs_with_era5")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest ERA5 results into DuckDB (schema era5)",
    )
    add_db_arg(parser)
    parser.add_argument(
        "--temperature-csv",
        type=Path,
        default=None,
        help="TopoSCALE output CSV (pixel_id, date, hour, T_pixel_C, ...)",
    )
    parser.add_argument(
        "--era5land-csv",
        type=Path,
        default=None,
        help="ERA5-Land precipitation CSV (pixel_id, date, total_precip_m, snowfall_m)",
    )
    parser.add_argument(
        "--partition-csv",
        type=Path,
        default=None,
        help="Rain/snow partition CSV (pixel_id, date, threshold_C, snow_corrected_m, ...)",
    )
    parser.add_argument(
        "--dewpoint-csv",
        type=Path,
        default=None,
        help="ERA5-Land dewpoint CSV (pixel_id, date, dewpoint_2m_K)",
    )
    args = parser.parse_args()

    ingest_era5(
        db_path=args.db,
        temperature_csv=args.temperature_csv,
        era5land_csv=args.era5land_csv,
        partition_csv=args.partition_csv,
        dewpoint_csv=args.dewpoint_csv,
    )


if __name__ == "__main__":
    main()
