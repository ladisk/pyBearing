import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../")
sys.path.insert(0, "../..")

import pytest

from pybearing.analysis.fault_frequencies_analysis import *
from pybearing.core.bearing import Bearing
from pybearing.core.fault_frequencies import FaultFrequencies
from pybearing.data.bearing_database import BearingDatabase


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

VALID_FAULT_FREQUENCIES = {
    "f_cage": 0.3683,
    "f_rolling_element_about_axis": 1.7661,
    "f_outer_ring": 2.5779,
    "f_rolling_element": 3.5322,
    "f_inner_ring": 4.4221,
    "f_of_rotation": 1.0
}

class TestFrequencyCage:
    def test_eq_f_cage(self):
        bearing = Bearing(**VALID_BEARING)
        f_c = eq_f_cage(bearing)

        assert f_c == pytest.approx(VALID_FAULT_FREQUENCIES["f_cage"], rel=1e-6)

    def test_eq_f_cage_outer_rotating(self):
        bearing = Bearing(**VALID_BEARING)
        f_c = eq_f_cage(bearing, inner_rotating=False)

        assert f_c == pytest.approx(1-VALID_FAULT_FREQUENCIES["f_cage"], rel=1e-6)

    def test_eq_f_cage_frequency_of_rotation(self):
        bearing = Bearing(**VALID_BEARING)
        f_c = eq_f_cage(bearing, frequency_of_rotation=3.0)

        result = round(0.5 * 3.0 * (1 - (2.770 / 10.514) * 1.0), 4)

        assert f_c == pytest.approx(result, rel=1e-6)

class TestFrequencyRollingElementAboutAxis:
    def test_eq_f_rolling_element_about_axis(self):
        bearing = Bearing(**VALID_BEARING)
        f_re_ax = eq_f_rolling_element_about_axis(bearing)

        assert f_re_ax == pytest.approx(VALID_FAULT_FREQUENCIES["f_rolling_element_about_axis"], rel=1e-6)

    def test_eq_f_rolling_element_about_axis_frequency_of_rotation(self):
        bearing = Bearing(**VALID_BEARING)
        f_re_ax = eq_f_rolling_element_about_axis(bearing, frequency_of_rotation=3.0)

        result = round(0.5 * (10.514/2.770) * 3.0 * (1 - (2.770**2/10.514**2)*1.0), 4)

        assert f_re_ax == pytest.approx(result, rel=1e-6)

class TestFrequencyOuterRing:
    def test_eq_f_outer_ring(self):
        bearing = Bearing(**VALID_BEARING)
        f_o = eq_f_outer_ring(bearing)

        assert f_o == pytest.approx(VALID_FAULT_FREQUENCIES["f_outer_ring"], rel=1e-6)

    def test_eq_f_outer_ring_frequency_of_rotation(self):
        bearing = Bearing(**VALID_BEARING)
        f_o = eq_f_outer_ring(bearing, frequency_of_rotation=3.0)

        result = round(0.5 * 7 * 3.0 * (1 - (2.770/10.514)*1.0), 4)

        assert f_o == pytest.approx(result, rel=1e-6)

class TestFrequencyRollingElement:
    def test_eq_f_rolling_element(self):
        bearing = Bearing(**VALID_BEARING)
        f_re = eq_f_rolling_element(bearing)

        assert f_re == pytest.approx(VALID_FAULT_FREQUENCIES["f_rolling_element"], rel=1e-6)

    def test_eq_f_rolling_element_frequency_of_rotation(self):
        bearing = Bearing(**VALID_BEARING)
        f_re = eq_f_rolling_element(bearing, frequency_of_rotation=3.0)

        f_re_ax = eq_f_rolling_element_about_axis(bearing, frequency_of_rotation=3.0)
        result = round(2 * f_re_ax, 4)

        assert f_re == pytest.approx(result, rel=1e-6)

class TestFrequencyInnerRing:
    def test_eq_f_inner_ring(self):
        bearing = Bearing(**VALID_BEARING)
        f_ir = eq_f_inner_ring(bearing)

        assert f_ir == pytest.approx(VALID_FAULT_FREQUENCIES["f_inner_ring"], rel=1e-6)

    def test_eq_f_inner_ring_frequency_of_rotation(self):
        bearing = Bearing(**VALID_BEARING)
        f_ir = eq_f_inner_ring(bearing, frequency_of_rotation=3.0)

        result = round(0.5 * 7 * 3.0 * (1 + (2.770/10.514)*1.0), 4)

        assert f_ir == pytest.approx(result, rel=1e-6)

class TestGetFaultFrequencies:
    def test_get_fault_frequencies_geometry(self):
        bearing = Bearing(**VALID_BEARING)
        frequencies = get_fault_frequencies(bearing, rpm=60.0, inner_rotating=True, method="geometry")

        assert frequencies == FaultFrequencies(**VALID_FAULT_FREQUENCIES)

    def test_get_fault_frequencies_database(self):
        bearing = Bearing(**VALID_BEARING)
        frequencies = get_fault_frequencies(bearing, rpm=60.0, inner_rotating=True, method="database")

        assert frequencies == FaultFrequencies(**VALID_FAULT_FREQUENCIES)

    def test_calculate_from_geometry(self):
        bearing = Bearing(**VALID_BEARING)
        frequencies = calculate_from_geometry(bearing, rpm=60.0, inner_rotating=True)

        assert frequencies == FaultFrequencies(**VALID_FAULT_FREQUENCIES)

    def test_calculate_from_database(self):
        bearing = Bearing(**VALID_BEARING)
        database = BearingDatabase()
        frequencies = calculate_from_database(bearing, rpm=60.0, inner_rotating=True, database=database)

        assert frequencies == FaultFrequencies(**VALID_FAULT_FREQUENCIES)
