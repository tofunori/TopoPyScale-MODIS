# TopoPyScale-MODIS

Lightweight [TopoSCALE](https://doi.org/10.5194/gmd-7-387-2014) downscaling for MODIS glacier pixels, inspired by [TopoPyScale](https://github.com/ArcticSnow/TopoPyScale) (Filhol & Fiddes, 2023).

Interpolates ERA5 pressure-level data (temperature, humidity, pressure) to individual glacier pixel elevations, then partitions precipitation into rain/snow using the [Jennings et al. (2018)](https://doi.org/10.1038/s41467-018-03629-7) logistic model.

Designed for the MODIS glacier albedo pipeline but usable for any pixel-scale ERA5 downscaling task.

## Install

```bash
pip install topopyscale-modis

# With DuckDB ingest support:
pip install "topopyscale-modis[ingest]"

# With ARCO-ERA5 (Google Cloud Zarr) support:
pip install "topopyscale-modis[arco]"

# Development:
pip install -e ".[dev]"
```

## Quick Start (CLI)

```bash
# 1. Download ERA5 pressure-level data (T + Z + q)
era5-download --region 02-03 --year 2023 --months 06,07,08 --out-dir /db/era5

# 2. Interpolate to pixel elevations (TopoSCALE)
era5-interpolate --era5-nc /db/era5/era5_pressure_02-03_2023_06_07_08.nc \
    --pixels-csv pixels.csv --out-csv toposcale.csv --workers 8

# 3. Partition precipitation into rain/snow (Jennings logistic)
era5-partition --temperature-csv toposcale.csv --era5land-csv precip.csv \
    --jennings-logistic --out-csv partition.csv

# 4. Ingest into DuckDB
era5-ingest --db modis.duckdb --temperature-csv toposcale.csv --partition-csv partition.csv
```

## Python API

```python
import pandas as pd
from topopyscale_modis import interpolate_temperature, partition_jennings_logistic

# Define pixel locations
pixels = pd.DataFrame({
    "pixel_id": [1, 2, 3],
    "elevation": [2500, 3000, 3500],
    "latitude": [52.0, 52.5, 53.0],
    "longitude": [-118.0, -117.5, -117.0],
})

# Interpolate ERA5 to pixel elevations
temp_df = interpolate_temperature("era5_pressure.nc", pixels, workers=4)

# Partition precipitation (requires humidity in ERA5 NetCDF)
result = partition_jennings_logistic(temp_df, precip_df)
```

## Custom Regions

```python
from topopyscale_modis import register_region, get_region_bbox

register_region("03-01", "Alps_Central", [48, 5, 44, 16])
bbox = get_region_bbox("03-01")  # [48, 5, 44, 16]
```

## Jennings (2018) Rain/Snow Partition

Three methods are available:

| Method | Model | Inputs |
|--------|-------|--------|
| `partition_rain_snow()` | Fixed thresholds | T only |
| `partition_jennings()` | RH-adaptive T50 | T + surface RH |
| `partition_jennings_logistic()` | Full logistic | T + RH + P at pixel elevation |

The **logistic model** (recommended) uses coefficients from Jennings et al. (2018) Supplementary Table 2:

- **Bivariate**: p(snow) = 1 / (1 + exp(-10.04 + 1.41*T + 0.09*RH))
- **Trivariate**: p(snow) = 1 / (1 + exp(-12.80 + 1.41*T + 0.09*RH + 0.03*P))

where T is in Celsius, RH in %, and P in kPa.

## References

If you use this package, please cite:

```bibtex
@article{fiddes2014toposcale,
  title={TopoSCALE v.1.0: downscaling gridded climate data in complex terrain},
  author={Fiddes, Joel and Gruber, Stephan},
  journal={Geoscientific Model Development},
  volume={7},
  pages={387--405},
  year={2014},
  doi={10.5194/gmd-7-387-2014}
}

@article{filhol2023topopyscale,
  title={TopoPyScale: A Python Package for Hillslope Climate Downscaling},
  author={Filhol, Simon and Fiddes, Joel and Aalstad, Kristoffer},
  journal={Journal of Open Source Software},
  volume={8},
  number={86},
  pages={5059},
  year={2023},
  doi={10.21105/joss.05059}
}

@article{jennings2018spatial,
  title={Spatial variation of the rain-snow temperature threshold across the Northern Hemisphere},
  author={Jennings, Keith S. and Winchell, Taylor S. and Livneh, Ben and Molotch, Noah P.},
  journal={Nature Communications},
  volume={9},
  pages={1148},
  year={2018},
  doi={10.1038/s41467-018-03629-7}
}
```

## License

MIT
