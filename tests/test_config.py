"""Tests for topopyscale_modis.config."""

import pytest

from topopyscale_modis.config import (
    DEFAULT_THRESHOLDS_C,
    PRESSURE_LEVELS_HPA,
    REGION_BBOX,
    RGI_REGIONS,
    G,
    get_region_bbox,
    normalize_region,
    register_region,
    validate_region,
)


class TestConfigImports:
    def test_rgi_regions_present(self):
        assert "02-03" in RGI_REGIONS

    def test_pressure_levels(self):
        assert len(PRESSURE_LEVELS_HPA) == 16
        assert PRESSURE_LEVELS_HPA[0] == 500
        assert PRESSURE_LEVELS_HPA[-1] == 1000

    def test_validate_region_ok(self):
        assert validate_region("02-03") == "02-03"
        assert validate_region("02_03") == "02-03"

    def test_validate_region_bad(self):
        with pytest.raises(ValueError, match="Unknown region"):
            validate_region("99-99")

    def test_normalize_region(self):
        assert normalize_region("02-03") == "02_03"

    def test_region_bbox(self):
        bbox = REGION_BBOX["02-03"]
        assert len(bbox) == 4
        assert bbox[0] > bbox[2]  # North > South

    def test_default_thresholds(self):
        assert DEFAULT_THRESHOLDS_C == [-1.0, 0.0, 1.0, 1.5, 2.0]

    def test_gravity_constant(self):
        assert G == pytest.approx(9.80665, abs=1e-5)

    def test_register_region(self):
        register_region("99-01", "Test_Region", [45, -75, 40, -70])
        assert "99-01" in RGI_REGIONS
        assert get_region_bbox("99-01") == [45, -75, 40, -70]
        assert validate_region("99-01") == "99-01"
        # Cleanup
        del RGI_REGIONS["99-01"]
        del REGION_BBOX["99-01"]

    def test_get_region_bbox_unknown(self):
        with pytest.raises(KeyError, match="Unknown region"):
            get_region_bbox("88-88")
