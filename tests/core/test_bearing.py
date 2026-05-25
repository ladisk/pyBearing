import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../")
sys.path.insert(0, "../..")

import pytest

from pybearing.core.bearing import *


VALID_BEARING = {
    "name": "NMB_1560kk",
    "d": 6.0,
    "D": 15.0,
    "B": 5.0,
    "D_b": 2.770,
    "D_m": 10.514,
    "theta": 0.0,
    "N_b": 7,
    "bearing_type": "Deep Groove Ball Bearing",
    "manufacturer": "NMB",
}


class TestBearing:
    def test_bearing_creation(self):
        bearing = Bearing(**VALID_BEARING)

        assert bearing.name == VALID_BEARING["name"]
        assert bearing.d == VALID_BEARING["d"]
        assert bearing.D == VALID_BEARING["D"]
        assert bearing.B == VALID_BEARING["B"]

    @pytest.mark.parametrize("missing_field", ["name", "d", "D", "B"])
    def test_required_fields(self, missing_field):
        data = VALID_BEARING.copy()
        data.pop(missing_field)

        with pytest.raises(TypeError):
            Bearing(**data)

    def test_invalid_D_b(self):
        data = VALID_BEARING.copy()
        data["D_b"] = -1

        with pytest.raises(ValueError, match="D_b must be positive"):
            Bearing(**data)

    def test_invalid_D_m(self):
        data = VALID_BEARING.copy()
        data["D_m"] = -1

        with pytest.raises(ValueError, match="D_m must be positive"):
            Bearing(**data)

    def test_invalid_theta(self):
        data = VALID_BEARING.copy()
        data["theta"] = 2.0

        with pytest.raises(ValueError, match="theta must be between 0 and pi/2"):
            Bearing(**data)

    def test_invalid_N_b(self):
        data = VALID_BEARING.copy()
        data["N_b"] = 2

        with pytest.raises(ValueError, match="N_b must be greater than 2"):
            Bearing(**data)

    def test_bearing_is_frozen(self):
        bearing = Bearing(**VALID_BEARING)

        with pytest.raises(AttributeError):
            bearing.d = 10.0

    def test_internal_geometry_specified(self):
        bearing = Bearing(**VALID_BEARING)
        assert bearing.internal_geometry_specified is True

        # Create a bearing without internal geometry parameters
        bearing_no_geometry = Bearing(name="Bearing 2", d=10.0, D=20.0, B=5.0)
        assert bearing_no_geometry.internal_geometry_specified is False


class TestBearingFromDict:
    def test_from_dict_creates_bearing(self):
        bearing = Bearing.from_dict(VALID_BEARING)

        assert isinstance(bearing, Bearing)
        assert bearing.name == VALID_BEARING["name"]
        assert bearing.d == VALID_BEARING["d"]

    def test_from_dict_ignores_extra_keys(self):
        data = VALID_BEARING.copy()
        data["extra_key"] = "extra_value"

        bearing = Bearing.from_dict(data)

        assert isinstance(bearing, Bearing)
        assert not hasattr(bearing, "extra_key")

    def test_from_dict_missing_required_key(self):
        data = VALID_BEARING.copy()
        data.pop("name")

        with pytest.raises(TypeError):
            Bearing.from_dict(data)

    def test_from_dict_invalid_values(self):
        data = VALID_BEARING.copy()
        data["D_b"] = -1

        with pytest.raises(ValueError, match="D_b must be positive"):
            Bearing.from_dict(data)
