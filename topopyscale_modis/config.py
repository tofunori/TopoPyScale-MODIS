"""Constants and configuration for the ERA5 pipeline.

Provides pressure levels, bounding boxes, default thresholds,
and RGI region metadata with a public API for adding custom regions.
"""

from __future__ import annotations

# === ERA5 pressure levels (hPa), 500-1000 ===
PRESSURE_LEVELS_HPA: list[int] = [
    500, 550, 600, 650, 700, 750, 775, 800,
    825, 850, 875, 900, 925, 950, 975, 1000,
]

# === CDS variables to download ===
CDS_VARIABLES: list[str] = ["temperature", "geopotential", "specific_humidity"]

# === Bounding boxes per region [North, West, South, East] with +1 degree margin ===
REGION_BBOX: dict[str, list[float]] = {
    "01-01": [70, -157, 66, -140],
    "01-02": [64, -160, 58, -140],
    "01-03": [60, -177, 51, -151],
    "01-04": [63, -152, 58, -143],
    "01-05": [63, -145, 58, -136],
    "01-06": [61, -139, 53, -125],
    "02-01": [66, -134, 59, -125],
    "02-02": [57, -130, 48, -120],
    "02-03": [59, -129, 48, -114],
    "02-04": [50, -124, 35, -117],
    "02-05": [50, -118, 36, -104],
}

# === Rain/snow partition thresholds (degrees C) ===
DEFAULT_THRESHOLDS_C: list[float] = [-1.0, 0.0, 1.0, 1.5, 2.0]

# === Gravity constant for geopotential -> altitude conversion ===
G = 9.80665

# === RGI regions ===
RGI_REGIONS: dict[str, str] = {
    "01-01": "Brooks_Range",
    "01-02": "Alaska_Range",
    "01-03": "Alaska_Peninsula",
    "01-04": "Chugach_Kenai",
    "01-05": "St_Elias_Glacier_Bay",
    "01-06": "Coast_Mountains_Alaska",
    "02-01": "Northern_BC_Yukon",
    "02-02": "Coast_Mountains_BC",
    "02-03": "Canadian_Rockies",
    "02-04": "Cascade_Sierra_Nevada",
    "02-05": "US_Rockies",
}


def register_region(code: str, name: str, bbox: list[float]) -> None:
    """Register a custom region for use with the pipeline.

    Parameters
    ----------
    code : str
        Region code in dash format (e.g. "03-01").
    name : str
        Human-readable region name.
    bbox : list[float]
        Bounding box as [North, West, South, East] in degrees.
    """
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements [N, W, S, E], got {len(bbox)}")
    dash = code.strip().replace("_", "-")
    RGI_REGIONS[dash] = name
    REGION_BBOX[dash] = bbox


def get_region_bbox(region: str) -> list[float]:
    """Return the bounding box [N, W, S, E] for a region code.

    Raises KeyError if the region is not registered.
    """
    dash = region.strip().replace("_", "-")
    if dash not in REGION_BBOX:
        raise KeyError(f"Unknown region: {region!r}. Register it with register_region().")
    return REGION_BBOX[dash]


def normalize_region(region: str) -> str:
    """Normalize region code: '02-03' -> '02_03'."""
    return region.strip().replace("-", "_")


def validate_region(region: str) -> str:
    """Validate a region code and return its dash-separated form.

    Raises ValueError if the region is unknown.
    """
    dash = region.strip().replace("_", "-")
    if dash not in RGI_REGIONS:
        raise ValueError(f"Unknown region: {region!r}. Valid: {sorted(RGI_REGIONS)}")
    return dash
