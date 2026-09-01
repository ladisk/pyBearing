import sys
sys.path.insert(0, ".")
sys.path.insert(0, "../")
sys.path.insert(0, "../..")

import pytest
import numpy as np
import pandas as pd

from pybearing.analysis.envelope_analysis import *
from pybearing.analysis.filter_search_results import FilterSearchResults
from pybearing.data.bearing_database import BearingDatabase


class TestEnvelopeAnalysis:
    def test_envelope_extraction(self):
        path_to_raw_signal = "tests/test_data/raw_force.npy"
        path_to_envelope_signal = "tests/test_data/envelope_force.npy"

        raw_signal = np.load(path_to_raw_signal) # time, force
        envelope_signal = np.load(path_to_envelope_signal) # time, envelope

        fs = 1 / (raw_signal[0, 1] - raw_signal[0, 0]) # sampling frequency
        envelope = envelope_extraction(raw_signal[1], fs, [100, 200])
        assert np.allclose(envelope, envelope_signal[1], rtol=1e-6)

    def test_rms_around_fault_frequencies_of_envelope_psd(self):
        path_to_envelope_signal = "tests/test_data/envelope_force.npy"

        envelope_signal = np.load(path_to_envelope_signal) # time, envelope

        true_result = np.array([0.01601943, 0.05209416, 0.13242861, 0.04278608, 0.12879152])

        database = BearingDatabase()
        characteristic_fault_frequencies = database.find_characteristic_frequencies_by_bearing_name("NMB_1560kk")[0]
        characteristic_fault_frequencies_scaled = characteristic_fault_frequencies * 5.1

        result = rms_around_fault_frequencies_of_envelope(envelope_signal[1], 6400, characteristic_fault_frequencies_scaled, min_margin=0.05, min_samples=2, rms_from="psd")
        result_values =  np.array(list(result.values()))

        assert np.allclose(result_values, true_result, rtol=1e-6)

    def test_rms_around_fault_frequencies_of_envelope_signal(self):
        path_to_envelope_signal = "tests/test_data/envelope_force.npy"

        envelope_signal = np.load(path_to_envelope_signal) # time, envelope

        true_result = np.array([0.07057886, 0.04013453, 0.12742358, 0.03425645, 0.12383159])

        database = BearingDatabase()
        characteristic_fault_frequencies = database.find_characteristic_frequencies_by_bearing_name("NMB_1560kk")[0]
        characteristic_fault_frequencies_scaled = characteristic_fault_frequencies * 5.1

        result = rms_around_fault_frequencies_of_envelope(envelope_signal[1], 6400, characteristic_fault_frequencies_scaled, min_margin=0.05, min_width=0.5, rms_from="signal")
        result_values =  np.array(list(result.values()))

        assert np.allclose(result_values, true_result, rtol=1e-6)

    def test_best_filter_for_envelope_extraction_uses_default_objective_function(self):
        result_1 = FilterSearchResults(
            level=np.array([1, 2]),
            f_low=np.array([10.0, 10.0]),
            f_high=np.array([20.0, 25.0]),
            harmonics_score=PerFaultResults(
                cage=np.array([1.0, 3.0]),
                rolling_element_about_axis=np.array([1.0, 3.0]),
                outer_ring=np.array([1.0, 3.0]),
                rolling_element=np.array([1.0, 3.0]),
                inner_ring=np.array([1.0, 3.0]),
            )
        )
        result_2 = FilterSearchResults(
            level=np.array([1, 2]),
            f_low=np.array([10.0, 10.0]),
            f_high=np.array([20.0, 25.0]),
            harmonics_score=PerFaultResults(
                cage=np.array([2.0, 4.0]),
                rolling_element_about_axis=np.array([2.0, 4.0]),
                outer_ring=np.array([2.0, 4.0]),
                rolling_element=np.array([2.0, 4.0]),
                inner_ring=np.array([2.0, 4.0]),
            )
        )

        results = best_filter_for_envelope_extraction(
            [result_1, result_2],
            ["signal_1", "signal_2"],
            evaluate_harmonics_score=True
        )

        assert isinstance(results["summary_harmonics_score"], pd.DataFrame)
        assert isinstance(results["selection_results_harmonics_score"], pd.DataFrame)

    def test_best_filter_for_envelope_extraction_accepts_custom_objective_function(self):
        result_1 = FilterSearchResults(
            level=np.array([1, 2]),
            f_low=np.array([10.0, 10.0]),
            f_high=np.array([20.0, 25.0]),
            harmonics_score=PerFaultResults(
                cage=np.array([1.0, 3.0]),
                rolling_element_about_axis=np.array([1.0, 3.0]),
                outer_ring=np.array([1.0, 3.0]),
                rolling_element=np.array([1.0, 3.0]),
                inner_ring=np.array([1.0, 3.0]),
            )
        )
        result_2 = FilterSearchResults(
            level=np.array([1, 2]),
            f_low=np.array([10.0, 10.0]),
            f_high=np.array([20.0, 25.0]),
            harmonics_score=PerFaultResults(
                cage=np.array([2.0, 4.0]),
                rolling_element_about_axis=np.array([2.0, 4.0]),
                outer_ring=np.array([2.0, 4.0]),
                rolling_element=np.array([2.0, 4.0]),
                inner_ring=np.array([2.0, 4.0]),
            )
        )

        results = best_filter_for_envelope_extraction(
            [result_1, result_2],
            ["signal_1", "signal_2"],
            evaluate_harmonics_score=True,
            objective_function=lambda mean, std: np.log(np.clip(mean, 1e-9, None)) + 0.5 * std,
        )

        assert isinstance(results["summary_harmonics_score"], pd.DataFrame)
        assert isinstance(results["selection_results_harmonics_score"], pd.DataFrame)
