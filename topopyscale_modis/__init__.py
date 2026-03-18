"""TopoPyScale-MODIS: lightweight TopoSCALE downscaling for MODIS glacier pixels.

Implements the Fiddes & Gruber (2014) TopoSCALE method for interpolating ERA5
pressure-level data to individual glacier pixel elevations, plus Jennings et al.
(2018) rain/snow partitioning.
"""

__version__ = "0.1.0"

from topopyscale_modis.config import get_region_bbox as get_region_bbox
from topopyscale_modis.config import register_region as register_region
from topopyscale_modis.partition import partition_jennings_logistic as partition_jennings_logistic
from topopyscale_modis.partition import partition_rain_snow as partition_rain_snow
from topopyscale_modis.toposcale import interpolate_temperature as interpolate_temperature
