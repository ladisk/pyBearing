import numpy as np
from scipy.signal import hilbert

from ..signal.features import rms_from_signal, rms_from_psd
from ..signal.filtering import filter_signal
from ..signal.spectral import power_spectral_density
from ..core.fault_frequencies import FaultFrequencies
from ..visualization.envelope_analysis import plot_envelope


def envelope_extraction(x:np.ndarray, fs:int, cutoff, filter_type:str="bandpass", order:int=2, plot:bool=False) -> np.ndarray:
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

def rms_around_fault_frequencies_of_envelope(x:np.ndarray, fs:int, fault_frequencies:FaultFrequencies, min_margin:float = 0.05, min_samples:int = 2, min_width:float = 1.0, rms_from:str="psd") -> dict:
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

def best_parameters_for_envelope_extraction():
    pass
