import numpy as np
from scipy.signal import hilbert
from scipy.stats import kurtosis

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

def filter_search_for_envelope_extraction(x:np.ndarray,
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
                                          calculate_kurtosis_score:bool = False) -> dict:
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
    dict
        A dictionary containing the:
        - "level", "f_low", and "f_high" for each evaluated frequency band.
        - "harmonics_results" indicating whether the harmonics score was calculated for each band.
        - "{fault_frequency}_harmonics_score" for each fault frequency and 
           each evaluated frequency band (if calculate_harmonics_score is True).
        - "kurtosis_results" indicating whether the kurtosis score was calculated for each band.
        - "kurtosis_score" for each evaluated frequency band (if calculate_kurtosis_score is True).
    """
    if max_frequency is None or max_frequency > fs/2:
        max_frequency = fs/2

    if min_frequency is None or min_frequency < 1:
        min_frequency = 1

    bands_container_harmonics = {}

    results = {
        "harmonics_results": calculate_harmonics_score,
        "kurtosis_results": calculate_kurtosis_score,
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

    return results

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
