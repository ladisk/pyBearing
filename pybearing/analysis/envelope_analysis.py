import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import kurtosis, zscore
from sklearn.model_selection import LeaveOneGroupOut

from .filter_search_results import PerFaultResults, FilterSearchResults
from ..signal.features import rms_from_signal, rms_from_psd
from ..signal.filtering import filter_signal
from ..signal.spectral import power_spectral_density
from ..core.fault_frequencies import FaultFrequencies
from ..visualization.envelope_analysis import plot_envelope


def envelope_extraction(x:np.ndarray,
                        fs:int,
                        cutoff,
                        filter_type:str="bandpass",
                        order:int=2,
                        plot:bool=False) -> np.ndarray:
    """
    Envelope extraction demodulates resonance vibrations to expose the underlying fault frequencies.
    It applies a bandpass filter to the input signal, computes the Hilbert transform to obtain the analytic signal,
    and then take the absolute value to get the envelope of the signal.

    parameters
    ----------
    x : np.ndarray
        1D signal from which to extract the envelope.
    fs : int
        Sampling frequency of the signal.
    cutoff : tuple
        Cutoff frequencies for the bandpass filter.
    filter_type : str
        Type of the bandpass filter.
    order : int, optional
        Order of the bandpass filter, by default 2.
    plot : bool, optional
        Whether to plot the original signal and the extracted envelope, by default False.

    returns
    -------
    np.ndarray
        Envelope of the input signal.
    """
    filtered_signal = filter_signal(x, fs, cutoff, filter_type, order)
    hilbert_transform = hilbert(filtered_signal)
    envelope = np.abs(hilbert_transform)

    if plot:
        plot_envelope(x, envelope, fs)

    return envelope

def rms_around_fault_frequencies_of_envelope(x:np.ndarray,
                                             fs:int,
                                             fault_frequencies:FaultFrequencies,
                                             min_margin:float = 0.05,
                                             min_samples:int = 2,
                                             min_width:float = 1.0,
                                             rms_from:str = "psd") -> dict:
    """
    Calculates the RMS values of the envelope of a signal around the characteristic fault frequencies. 
    The function can calculate the RMS values either from the power spectral density (PSD) or directly 
    from the signal.

    parameters
    ----------
    x : np.ndarray
        Input signal in time domain.
    fs : int
        Sampling frequency of the input signal.
    fault_frequencies : FaultFrequencies
        An instance of the FaultFrequencies class containing the characteristic fault frequencies.
    min_margin : float, optional
        Minimum margin around the fault frequencies to consider when calculating RMS values, by default 0.05 (5%).
    min_samples : int, optional
        Minimum number of frequency bins that must be within the margin for the RMS value to be considered valid, by default 2.
    min_width : float, optional
        Minimum width of the frequency band around the fault frequencies to consider when calculating RMS values, by default 1.0 Hz.
    rms_from : str, optional
        Method to calculate RMS values. Can be either "psd" to calculate from power spectral density or "signal" to calculate directly from the signal, by default "psd".

    returns
    -------
    dict
        A dictionary containing the RMS values around each characteristic fault frequency, with keys in the format "rms_{fault_type}".
    """
    fault_freqs = fault_frequencies.fault_frequencies()
    results = {}

    if rms_from == "psd":
        freq, Pxx = power_spectral_density(x, fs, nperseg=len(x))
        for key, value in fault_freqs.items():
            is_ok = False
            margin = min_margin
            while not is_ok:
                lower_bound = (1 - margin) * value
                upper_bound = (1 + margin) * value
                count = sum((freq > lower_bound) & (freq < upper_bound))
                if count >= min_samples:
                    is_ok = True
                else:
                    margin += 0.01
            lower_bound = (1 - margin) * value
            upper_bound = (1 + margin) * value
            rms_value = rms_from_psd(freq, Pxx, freq_bands=(lower_bound, upper_bound))
            results[f"rms_{key}"] = rms_value

    elif rms_from == "signal":
        for key, value in fault_freqs.items():
            is_ok = False
            margin = min_margin
            lower_bound = max(min((1 - margin) * value, value - min_width / 2), 0.01)
            upper_bound = min(max((1 + margin) * value, value + min_width / 2), fs/2 - 1)

            rms_value = rms_from_signal(x, fs, freq_bands=(lower_bound, upper_bound))
            results[f"rms_{key}"] = rms_value
    else:
        raise ValueError("Invalid value for rms_from. Expected 'psd' or 'signal'.")
    
    return results

def filter_search_for_envelope_extraction(
        x:np.ndarray,
        fs:int,
        fault_frequencies:FaultFrequencies,
        max_level:int,
        compute_triadic:bool = False,
        max_frequency:float = None,
        min_frequency:float = None,
        epsilon:float = 1e-3,
        calculate_harmonics_score:bool = True,
        n_harmonics:int = 1,
        min_samples:int = 2,
        absolute_error:float = 1.0,
        relative_error:float = 0.0,
        calculate_kurtosis_score:bool = False
    ) -> FilterSearchResults:
    """
    Perform a search over different frequency bands for envelope extraction and calculate scores based on:
    1) The energy around the harmonics of interest compared to the energy in the rest of the spectrum for 
    each band and each fault frequency (if calculate_harmonics_score is True).
    2) The kurtosis of the filtered signal in each band (if calculate_kurtosis_score is True).

    parameters
    ----------
    x : np.ndarray
        Input signal in time domain.
    fs : int
        Sampling frequency of the input signal.
    fault_frequencies : FaultFrequencies
        An instance of the FaultFrequencies class containing the characteristic fault frequencies.
    max_level : int
        Maximum level of the search, which determines the number of frequency bands to evaluate.
    compute_triadic : bool, optional
        Whether to compute triadic bandwidths in addition to diadic bandwidths, by default False.
    max_frequency : float, optional
        Maximum frequency to consider in the search, by default None (which means fs/2).
    min_frequency : float, optional
        Minimum frequency to consider in the search, by default None (which means 1 Hz).
    epsilon : float, optional
        Small value to avoid numerical issues when defining frequency bands, by default 1e-3.
    calculate_harmonics_score : bool, optional    
        Whether to calculate the harmonics score based on the energy around the harmonics of interest, by default True.
    n_harmonics : int, optional
        The number of harmonics to consider when calculating the harmonics score, by default 1.
    min_samples : int, optional
        Minimum number of frequency bins that must be within the harmonic bands for the harmonics score to be considered
        valid, by default 2.
    absolute_error : float, optional
        Absolute error tolerance in Hz for determining the harmonic bands when calculating the harmonics score,
        by default 1.0 Hz.
    relative_error : float, optional
        Relative error tolerance for determining the harmonic bands when calculating the harmonics score,
        by default 0.0.
    calculate_kurtosis_score : bool, optional
        Whether to calculate the kurtosis score based on the kurtosis of the filtered signal in each band, by default False.

    returns
    -------
    FilterSearchResults
        An instance of the FilterSearchResults data class containing the:
        - "level", "f_low", and "f_high" for each evaluated frequency band.
        - "harmonics_score" (HarmonicsResults, containing the score for each fault frequency for each evaluated frequency band) (if calculate_harmonics_score is True else None).
        - "kurtosis_score" containing the kurtosis score for each evaluated frequency band (if calculate_kurtosis_score is True else None).
    """
    if max_frequency is None or max_frequency > fs/2:
        max_frequency = fs/2

    if min_frequency is None or min_frequency < 1:
        min_frequency = 1

    bands_container_harmonics = {}

    results = {
        "f_low": [],
        "f_high": [],
        "level": []
    }

    # Add keys to results dictionary for each fault frequency
    # Calculate the frequency bins for harmonics and create masks for excluded harmonics
    freq = np.fft.rfftfreq(len(x), 1/fs)
    # Kurtosis score will be calculated as the kurtosis of the envelope in the given frequency band
    results["kurtosis_score"] = []
    # Harmonics score will be calculated as the ratio of energy around the harmonics of interest to the energy in the rest of the spectrum for each band and each fault frequency
    for fault_frequency, fault_frequency_value in fault_frequencies.fault_frequencies().items():
        # Harmonics score
        results[f"{fault_frequency}_harmonics_score"] = []

        bands, mask_with_excluded_harmonics = _determine_bands_for_harmonics(freq, fault_frequency_value, n_harmonics, min_samples, absolute_error, relative_error)
        bands_container_harmonics[fault_frequency] = bands
        bands_container_harmonics[f"{fault_frequency}_excluded"] = mask_with_excluded_harmonics

        # TODO: expand to also include calculate_sidebands_score around the each fault frequency as it can be more dominant than the harmonics themselves in some cases

    for level in range(max_level):
        # compute diadic bandwidth
        bandwidth_dyadic = 2 * (max_frequency - min_frequency) / (2 ** (level + 1))

        f_low_all_dyadic = np.arange(min_frequency, max_frequency, bandwidth_dyadic)
        for f_low_dyadic in f_low_all_dyadic:
            f_high_dyadic = np.min(np.array([f_low_dyadic + bandwidth_dyadic, fs/2 - epsilon*fs]))
            f_low_dyadic = np.max(np.array([f_low_dyadic, epsilon*fs, min_frequency]))

            results["level"].extend([level])
            results["f_low"].extend([f_low_dyadic])
            results["f_high"].extend([f_high_dyadic])

            filter_dyadic = [f_low_dyadic, f_high_dyadic]

            if calculate_harmonics_score:
                envelope = envelope_extraction(x, fs, filter_dyadic, "bandpass")
                freq, Pxx = power_spectral_density(envelope, fs, len(envelope))

                for fault_frequency, fault_frequency_value in fault_frequencies.fault_frequencies().items():
                    bands = bands_container_harmonics[fault_frequency]
                    mask_with_excluded_harmonics = bands_container_harmonics[f"{fault_frequency}_excluded"]

                    score_harmonics = _calculate_score_from_harmonics(freq, Pxx, bands, mask_with_excluded_harmonics)
                    results[f"{fault_frequency}_harmonics_score"].extend([score_harmonics])

            if calculate_kurtosis_score:
                filtered_signal = filter_signal(x, fs, filter_dyadic, "bandpass")
                kurtosis_score = kurtosis(filtered_signal)
                results["kurtosis_score"].extend([kurtosis_score])

        # compute triadic bandwidth
        if compute_triadic:
            bandwidth_triadic = 2 * (max_frequency - min_frequency) / (3 * 2 ** (level + 1))

            f_low_all_triadic = np.arange(min_frequency, max_frequency, bandwidth_triadic)
            for f_low_triadic in f_low_all_triadic:
                f_high_triadic = np.min(np.array([f_low_triadic + bandwidth_triadic, fs/2 - epsilon*fs]))
                f_low_triadic = np.max(np.array([f_low_triadic, epsilon*fs, min_frequency]))

                results["level"].extend([np.round(level + np.log2(3) - 1, 3)])
                results["f_low"].extend([f_low_triadic])
                results["f_high"].extend([f_high_triadic])

                filter_triadic = [f_low_triadic, f_high_triadic]

                if calculate_harmonics_score:
                    envelope = envelope_extraction(x, fs, filter_triadic, "bandpass")
                    freq, Pxx = power_spectral_density(envelope, fs, len(envelope))

                    for fault_frequency, fault_frequency_value in fault_frequencies.fault_frequencies().items():
                        bands = bands_container_harmonics[fault_frequency]
                        mask_with_excluded_harmonics = bands_container_harmonics[f"{fault_frequency}_excluded"]

                        score_harmonics = _calculate_score_from_harmonics(freq, Pxx, bands, mask_with_excluded_harmonics)
                        results[f"{fault_frequency}_harmonics_score"].extend([score_harmonics])
                
                if calculate_kurtosis_score:
                    filtered_signal = filter_signal(x, fs, filter_triadic, "bandpass")
                    kurtosis_score = kurtosis(filtered_signal)
                    results["kurtosis_score"].extend([kurtosis_score])

    harmonics_results = PerFaultResults(
        cage = np.array(results["f_cage_harmonics_score"]),
        rolling_element_about_axis = np.array(results["f_rolling_element_about_axis_harmonics_score"]),
        outer_ring = np.array(results["f_outer_ring_harmonics_score"]),
        rolling_element = np.array(results["f_rolling_element_harmonics_score"]),
        inner_ring = np.array(results["f_inner_ring_harmonics_score"]),
    ) if calculate_harmonics_score else None

    filter_search_results = FilterSearchResults(
        level = np.array(results["level"]),
        f_low = np.array(results["f_low"]),
        f_high = np.array(results["f_high"]),
        harmonics_score = harmonics_results,
        kurtosis_score = np.array(results["kurtosis_score"]) if calculate_kurtosis_score else None,
    )

    return filter_search_results

def _determine_bands_for_harmonics(freq, fault_frequency:float, n_harmonics:int=1, min_samples:int = 2, absolute_error:float = 1.0, relative_error:float = 0.0):
    """
    Determine frequency bands around the harmonics of a given fault frequency, 
    ensuring that each band contains at least a minimum number of samples and 
    accounts for both absolute and relative error of harmonic occurrence.
    Bandwidth is determined by the maximum of absolute error, relative error, 
    and minimum samples requirement.

    parameters
    ----------
    freq: array-like
        Array of frequency values in frequency domain.
    fault_frequency: float
        The fundamental fault frequency.
    n_harmonics: int, optional
        The number of harmonics to consider, by default 1.
    min_samples: int, optional
        The minimum number of samples required in each band, by default 2.
    absolute_error: float, optional
        The absolute error tolerance, by default 1.0.
    relative_error: float, optional
        The relative error tolerance, by default 0.0.

    returns
    -------
    list of tuples
        A list of (f_low, f_high) tuples representing the frequency bands for each harmonic.
    numpy array
        A boolean mask indicating which frequencies in `freq` are outside the defined harmonic bands.
    """
    bands = []
    
    d_freq = freq[1]-freq[0]

    for harmonic in range(1, n_harmonics+1):
        target_frequency = harmonic * fault_frequency
        if target_frequency > freq[-1]:
            break
        
        # determine bandwidth based on absolute and relative error
        bandwidth = np.max(np.array([absolute_error, relative_error * target_frequency, min_samples * d_freq]))
        
        # determine low and high frequency for the band
        f_low = target_frequency - bandwidth/2
        f_high = target_frequency + bandwidth/2
             
        bands.append((f_low, f_high))

    mask_with_excluded_harmonics = np.ones_like(freq, dtype=bool)
    for band in bands:
        mask_with_excluded_harmonics &= ~((freq > band[0]) & (freq < band[1]))
    
    return bands, mask_with_excluded_harmonics

def _calculate_score_from_harmonics(freq:np.ndarray, Pxx:np.ndarray, bands:list[tuple], mask_with_excluded_harmonics:np.ndarray) -> float:
    """
    Calculate a score based on the energy around the harmonics of interest compared to the energy in the rest of the spectrum.

    equation: score_harmonics = (rms_around_harmonics / rms_excluded_harmonics) * log(1 + rms_around_harmonics)

    parameters
    ----------
    freq : np.ndarray
        Frequency array.
    Pxx : np.ndarray
        Power spectral density array.
    bands : list[tuple]
        List of frequency bands around the harmonics of interest.
    mask_with_excluded_harmonics : np.ndarray
        Boolean mask indicating the excluded harmonics.

    returns
    -------
    float
        The calculated score.
    """
    energy_aroung_harmonics = rms_from_psd(freq, Pxx, bands)**2
    rms_around_harmonics = np.sqrt(np.sum(energy_aroung_harmonics))

    energy_excluded_harmonics = rms_from_psd(freq[mask_with_excluded_harmonics], Pxx[mask_with_excluded_harmonics])**2
    rms_excluded_harmonics = np.sqrt(np.sum(energy_excluded_harmonics))

    score_harmonics = (rms_around_harmonics / rms_excluded_harmonics) * np.log(1 + rms_around_harmonics)
    return score_harmonics

def best_filter_for_envelope_extraction(
        x: list[FilterSearchResults],
        signal_names: list,
        evaluate_harmonics_score: bool = False,
        objective_function = None,
        evaluate_kurtosis_score: bool = False
    ):
    """
    Determine the best filter for envelope extraction based on the results of multiple filter searches.
    For each bearing-characteristic-fault one bandpass filter is selected that maximizes the objective function.
    
    List of FilterSearchResults should contain results of signals from different bearings, but under
    the same operating conditions (e.g., same speed, load, etc.) and preferentially the same fault type,
    as different faults may show different characteristics in the envelope spectrum.

    parameters
    ----------
    x: list[FilterSearchResults] 
        List of results obtained by the filter_search_for_envelope_extraction method.
    signal_names: list
        List of signal names. Index of each name should corespond to index of result in x.
    evaluate_harmonics_score: bool, optional
        Whether to evaluate the harmonics score for each fault characteristic frequency, by default False.
    objective_function: callable, optional
        Function used to aggregate ``mean`` and ``std`` into a single score. The callable must accept
        ``mean`` and ``std`` as NumPy arrays or pandas Series and return the objective values. By
        default, the objective is ``mean - 0.5 * std``. If using a log-based objective, clip or
        shift the values to avoid invalid inputs such as ``log(negative)``.
    evaluate_kurtosis_score: bool, optional
        Whether to evaluate the kurtosis score for each fault characteristic frequency, by default False.

    returns
    -------
    """
    if objective_function is None:
        def objective_function(mean, std):
            return mean - 0.5 * std

    elif not callable(objective_function):
        raise TypeError("objective_function must be callable and accept mean and std arguments.")
    # ------------------------------
    # Temporary varaibles to store data
    # ------------------------------
    harmonics_score_dictionary = {
        "signal": [],
        "fault_characteristic_frequency": [],
        "level": [],
        "f_low": [],
        "f_high": [],
        "score": []
    }
    fault_characteristic_frequencies = None

    # ------------------------------
    # Loop across results and transform different scores for further processing
    # ------------------------------
    for i, result in enumerate(x):
        for key in list(vars(result).keys()):
            if key.endswith("score"):
                score_attribute = getattr(result, key, None)

                # Prepare data for harmonics score processing
                if evaluate_harmonics_score and key == "harmonics_score" and score_attribute is not None:
                    attributes = list(vars(score_attribute).keys())
                    fault_characteristic_frequencies = attributes
                    for attribute in attributes:
                        score_value = getattr(score_attribute, attribute, None)
                        z_score = zscore(score_value)
                        number_of_computated_scores = len(z_score)

                        harmonics_score_dictionary["signal"].extend([signal_names[i]] * number_of_computated_scores)
                        harmonics_score_dictionary["fault_characteristic_frequency"].extend([attribute] * number_of_computated_scores)
                        harmonics_score_dictionary["level"].extend(result.level)
                        harmonics_score_dictionary["f_low"].extend(result.f_low)
                        harmonics_score_dictionary["f_high"].extend(result.f_high)
                        harmonics_score_dictionary["score"].extend(z_score)

                # Prepare data for kurtosis score processing
                if evaluate_kurtosis_score and key == "kurtosis_score" and score_attribute is not None:
                    # TODO: Implement kurtosis score processing
                    raise NotImplementedError("Kurtosis score processing is not implemented yet.")

    # ------------------------------
    # Process the collected scores to determine the best filter for envelope extraction
    # ------------------------------
    if evaluate_harmonics_score:
        summary_harmonics_score, selection_results_harmonics_score = _determine_best_filter_for_envelope_extraction_using_harmonics_score(
            harmonics_score_dictionary,
            fault_characteristic_frequencies,
            objective_function
        )

    if evaluate_kurtosis_score:
        # TODO: Implement kurtosis score evaluation
        raise NotImplementedError("Kurtosis score evaluation is not implemented yet.")

    # ------------------------------
    # Create a return dictionary to store the results of the best filter selection process
    # Save the results of the function to xlsx file if desired
    # ------------------------------
    results = {}

    if evaluate_harmonics_score:
        results["summary_harmonics_score"] = summary_harmonics_score
        results["selection_results_harmonics_score"] = selection_results_harmonics_score
    if evaluate_kurtosis_score:
        # TODO: Append kurtosis score results to return_list
        raise NotImplementedError("Kurtosis score results are not implemented yet.")

    return results

def _determine_best_filter_for_envelope_extraction_using_harmonics_score(
        harmonics_score_dictionary: dict,
        fault_characteristic_frequencies: list,
        objective_function: callable
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Determine the best filter for envelope extraction based on the harmonics score.

    parameters
    ----------
    harmonics_score_dictionary : dict
        Dictionary containing the harmonics scores to determine the best filter for envelope extraction.
    fault_characteristic_frequencies : list
        List of fault characteristic frequencies to evaluate.
    objective_function : callable
        Function used to aggregate ``mean`` and ``std`` into a single score. The callable must accept

    returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        summary_harmonics_score : pd.DataFrame
            Summary of the harmonics scores.
        selection_results_harmonics_score : pd.DataFrame
            Results of the cross-validation for filter selection process.
    """
    REQUIRED_KEYS_IN_harmonics_score_dictionary = ["signal", "fault_characteristic_frequency", "level", "f_low", "f_high", "score"]
    for key in harmonics_score_dictionary.keys():
        if key not in REQUIRED_KEYS_IN_harmonics_score_dictionary:
            raise ValueError(f"Missing required key '{key}' in harmonics_score_dictionary. Required keys are: {REQUIRED_KEYS_IN_harmonics_score_dictionary}")
    # ------------------------------
    # Determining the best filter for envelope extraction based on the harmonics score
    # ------------------------------
    dataframe = pd.DataFrame(harmonics_score_dictionary)

    logo = LeaveOneGroupOut() # Cross-validation strategy

    best_band_results = []

    for fault in fault_characteristic_frequencies: # Evaluate for each fault separately
        sub_dataframe = dataframe[dataframe["fault_characteristic_frequency"] == fault]
        for train_idx, test_idx in logo.split(
                sub_dataframe,
                groups=sub_dataframe["signal"]):

            train_df = sub_dataframe.iloc[train_idx]
            test_df = sub_dataframe.iloc[test_idx]

            objective = (
                train_df
                .groupby(["level","f_low","f_high"])["score"]
                .agg(["mean","std"])
                .reset_index()
            )

            objective["std"] = objective["std"].fillna(0)

            objective["objective"] = objective_function(
                objective["mean"],
                objective["std"]
            )

            best = (
                objective
                .sort_values("objective", ascending=False)
                .iloc[0]
            )

            score = test_df[
                (test_df.level == best.level)
                &
                (test_df.f_low == best.f_low)
                &
                (test_df.f_high == best.f_high)
            ]["score"].iloc[0]

            test_sorted = (
                test_df
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )

            rank = test_sorted[
                (test_sorted.level == best.level)
                &
                (test_sorted.f_low == best.f_low)
                &
                (test_sorted.f_high == best.f_high)
            ].index[0] + 1

            best_band_results.append({
                "signal": test_df["signal"].iloc[0],
                "fault": fault,
                "level": best.level,
                "f_low": best.f_low,
                "f_high": best.f_high,
                "test_score": score,
                "rank": rank
            })
    # Detailed results of selection -> each fault from every signal is cross-validated on band selected from all other signals
    # rank is the position of the selected band in the sorted list of scores for the test signal -> eg. rank 2 means that best
    # band for other signals is the second best for the test signal.
    selection_results_harmonics_score = pd.DataFrame(best_band_results) 

    # Create summary_harmonics_score of selection results based on harmonics score
    summary_harmonics_score = (
        selection_results_harmonics_score
        .groupby(["fault", "level", "f_low", "f_high"])
        .agg(
            mean_test_score=("test_score", "mean"),
            std_test_score=("test_score", "std"),
            mean_rank=("rank", "mean"),
            max_rank=("rank", "max"),
            n=("test_score", "size")
        )
        .reset_index()
    )

    n_signals = selection_results_harmonics_score.groupby("fault")["signal"].nunique()

    selection_frequency = (
        selection_results_harmonics_score
        .groupby(["fault","level","f_low","f_high"])
        .size()
        .rename("selection_count")
        .reset_index()
    )

    selection_frequency["selection_frequency"] = (
        selection_frequency.apply(
            lambda r: r.selection_count / n_signals[r.fault],
            axis=1
        )
    )

    summary_harmonics_score = summary_harmonics_score.merge(
        selection_frequency,
        on=["fault","level","f_low","f_high"]
    )

    summary_harmonics_score = summary_harmonics_score.sort_values(
        ["fault","selection_frequency","mean_test_score"],
        ascending=[True, False, False]
    )

    return summary_harmonics_score, selection_results_harmonics_score
