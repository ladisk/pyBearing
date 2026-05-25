import numpy as np
from scipy.signal import butter, sosfiltfilt


def filter_signal(x:np.ndarray, fs:int, cutoff, filter_type:str, order:int=2) -> np.ndarray:
    """
    Filter 1D signal using a zero-phase digital filter.

    parameters
    ----------
    x : np.ndarray
        1D signal to be filtered.
    fs : int
        The sampling frequency of the signal.
    cutoff : float or tuple
        The cutoff frequency for the filter.
    filter_type : str
        The type of filter to apply. Must be one of 'lowpass', 'highpass', 'bandpass', or 'bandstop'.
    order : int, optional
        The order of the filter. Default is 2.

    returns
    -------
    filtered_signal : np.ndarray
        The filtered signal.
    """
    nyquist = 0.5 * fs
    if isinstance(cutoff, tuple) or isinstance(cutoff, list) or isinstance(cutoff, np.ndarray):
        normal_cutoff = (cutoff[0] / nyquist, cutoff[1] / nyquist)
    elif isinstance(cutoff, float) or isinstance(cutoff, int):
        normal_cutoff = cutoff / nyquist
    else:
        raise ValueError("Cutoff frequency must be a float/int or a tuple/list/array of two floats.")
    
    sos = butter(order, normal_cutoff, btype=filter_type, analog=False, output='sos')
    return sosfiltfilt(sos, x)
