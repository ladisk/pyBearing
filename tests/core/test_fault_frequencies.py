import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../")
sys.path.insert(0, "../..")

import pytest

from pybearing.core.fault_frequencies import *


VALID_FAULT_FREQUENCIES = {
    "f_cage": 0.3683,
    "f_rolling_element_about_axis": 1.7661,
    "f_outer_ring": 2.5779,
    "f_rolling_element": 3.5322,
    "f_inner_ring": 4.4221,
    "f_of_rotation": 1.0
}

class TestFaultFrequencies:
    def test_fault_frequencies_creation(self):
        frequencies = FaultFrequencies(**VALID_FAULT_FREQUENCIES)

        assert frequencies.f_cage == VALID_FAULT_FREQUENCIES["f_cage"]
        assert frequencies.f_rolling_element_about_axis == VALID_FAULT_FREQUENCIES["f_rolling_element_about_axis"]
        assert frequencies.f_outer_ring == VALID_FAULT_FREQUENCIES["f_outer_ring"]
        assert frequencies.f_rolling_element == VALID_FAULT_FREQUENCIES["f_rolling_element"]
        assert frequencies.f_inner_ring == VALID_FAULT_FREQUENCIES["f_inner_ring"]
        assert frequencies.f_of_rotation == VALID_FAULT_FREQUENCIES["f_of_rotation"]

    @pytest.mark.parametrize("missing_field", ["f_cage", "f_rolling_element_about_axis", "f_outer_ring", "f_rolling_element", "f_inner_ring"])
    def test_required_fields(self, missing_field):
        data = VALID_FAULT_FREQUENCIES.copy()
        data.pop(missing_field)

        with pytest.raises(TypeError):
            FaultFrequencies(**data)

    def test_invalid_f_cage(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data["f_cage"] = -1

        with pytest.raises(ValueError, match="Cage frequency f_cage must be positive"):
            FaultFrequencies(**data)
    
    def test_invalid_f_rolling_element_about_axis(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data["f_rolling_element_about_axis"] = -1

        with pytest.raises(ValueError, match="Rolling element about axis frequency f_rolling_element_about_axis must be positive"):
            FaultFrequencies(**data)

    def test_invalid_f_outer_ring(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data["f_outer_ring"] = -1

        with pytest.raises(ValueError, match="Outer ring frequency f_outer_ring must be positive"):
            FaultFrequencies(**data)

    def test_invalid_f_rolling_element(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data["f_rolling_element"] = -1

        with pytest.raises(ValueError, match="Rolling element frequency f_rolling_element must be positive"):
            FaultFrequencies(**data)

    def test_invalid_f_inner_ring(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data["f_inner_ring"] = -1

        with pytest.raises(ValueError, match="Inner ring frequency f_inner_ring must be positive"):
            FaultFrequencies(**data)

    def test_fault_frequencies(self):
        frequencies = FaultFrequencies(**VALID_FAULT_FREQUENCIES)
        fault_freqs = frequencies.fault_frequencies()

        assert fault_freqs["f_cage"] == frequencies.f_cage
        assert fault_freqs["f_rolling_element_about_axis"] == frequencies.f_rolling_element_about_axis
        assert fault_freqs["f_outer_ring"] == frequencies.f_outer_ring
        assert fault_freqs["f_rolling_element"] == frequencies.f_rolling_element
        assert fault_freqs["f_inner_ring"] == frequencies.f_inner_ring


class TestFaultFrequenciesMultiplication:
    def test_multiplication(self):
        frequencies = FaultFrequencies(**VALID_FAULT_FREQUENCIES)
        scaled_frequencies = frequencies * 2

        assert scaled_frequencies.f_cage == pytest.approx(0.7366, rel=1e-6)
        assert scaled_frequencies.f_rolling_element_about_axis == pytest.approx(3.5322, rel=1e-6)
        assert scaled_frequencies.f_outer_ring == pytest.approx(5.1558, rel=1e-6)
        assert scaled_frequencies.f_rolling_element == pytest.approx(7.0644, rel=1e-6)
        assert scaled_frequencies.f_inner_ring == pytest.approx(8.8442, rel=1e-6)
        assert scaled_frequencies.f_of_rotation == pytest.approx(2.0, rel=1e-6)

    def test_left_multiplication(self):
        frequencies = FaultFrequencies(**VALID_FAULT_FREQUENCIES)
        scaled_frequencies = 2 * frequencies

        assert scaled_frequencies.f_cage == pytest.approx(0.7366, rel=1e-6)
        assert scaled_frequencies.f_rolling_element_about_axis == pytest.approx(3.5322, rel=1e-6)
        assert scaled_frequencies.f_outer_ring == pytest.approx(5.1558, rel=1e-6)
        assert scaled_frequencies.f_rolling_element == pytest.approx(7.0644, rel=1e-6)
        assert scaled_frequencies.f_inner_ring == pytest.approx(8.8442, rel=1e-6)
        assert scaled_frequencies.f_of_rotation == pytest.approx(2.0, rel=1e-6)

class TestFaultFrequenciesFromDictDatabase:
    def test_from_dict_database_creates_faultfrequencies(self):
        frequencies = FaultFrequencies.from_dict_database(VALID_FAULT_FREQUENCIES)

        assert isinstance(frequencies, FaultFrequencies)
        assert frequencies.f_cage == VALID_FAULT_FREQUENCIES["f_cage"]

    def test_from_dict_database_ignores_extra_keys(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data["extra_key"] = "extra_value"

        frequencies = FaultFrequencies.from_dict_database(data)

        assert isinstance(frequencies, FaultFrequencies)
        assert not hasattr(frequencies, "extra_key")

    def test_from_dict_database_missing_required_key(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data.pop("f_cage")

        with pytest.raises(TypeError):
            FaultFrequencies.from_dict_database(data)

    def test_from_dict_database_invalid_values(self):
        data = VALID_FAULT_FREQUENCIES.copy()
        data["f_cage"] = -1

        with pytest.raises(ValueError, match="Cage frequency f_cage must be positive"):
            FaultFrequencies.from_dict_database(data)
