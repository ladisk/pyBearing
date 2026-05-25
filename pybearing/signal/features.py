import numpy as np

from .filtering import filter_signal


def rms(x:np.ndarray) -> float:
    """
    Calculate the root mean square of a 1D signal.

    parameters
    ----------
    x : np.ndarray
        1D input signal.
    
    returns
    -------
    rms_value : float
        The root mean square of the input signal.
    """
    if x.ndim != 1:
        raise ValueError("Input signal must be a 1D array.")
    
    return np.sqrt(np.mean(np.square(x)))

def rms_from_signal(x:np.ndarray, fs:float, freq_bands=None, order:int=2) -> float | np.ndarray:
    """
    Filters a signal using a specified bandpass filter and calculates its RMS value.
    
    parameters
    ----------
    x : np.ndarray
        Input signal in time domain
    fs : float
        Sampling frequency in Hz
    freq_bands : tuple, list, np.ndarray, optional
        Frequency bands to calculate rms values for. If None, the rms value of the whole signal is returned.
        (equivalent to just using rms() function). If a tuple/list/array of length 2 is given, the rms value 
        in the given frequency band is returned. If a 2D array is given, the rms values of each frequency band 
        is returned. The default is None.
    order : int, optional
        Order of the filter. The default is 2. See order filter of filter_signal() function for more details.
        NOTE: the order of the filter is doubled when using filter_signal() function, so the actual order of the
        filter is 2*order.
    
    returns
    -------
    float | np.ndarray
        RMS value(s) of the signal in specified frequency bands.
    """
    if len(x) == 0:
        raise ValueError("Input array is empty.")
    if not isinstance(x, np.ndarray):
        raise TypeError("Input signal must be a numpy array.")
    if not isinstance(fs, (float, int)):
        raise TypeError("Sampling frequency must be a float or integer.")
    
    def rms_in_band(freq_band):
        filtered_signal = filter_signal(x, fs, freq_band, "bandpass", order)
        return rms(filtered_signal)

    if freq_bands is None:
        return rms(x)
    elif isinstance(freq_bands, (tuple, list, np.ndarray)):
        if np.ndim(freq_bands) == 1:
            if len(freq_bands) != 2:
                raise ValueError("Frequency bands must have length 2.")
            return rms_in_band(freq_bands)
        else:
            return np.array([rms_in_band(band) for band in freq_bands])
    else:
        raise ValueError("Frequency bands must be a tuple/list/array or a 2D array.")

def rms_from_psd(freq:np.ndarray, Pxx:np.ndarray, freq_bands=None, exclude_zero_frequency:bool=True) -> float | np.ndarray:
    """
    Calculates root mean square value from power spectral density.
    
    parameters
    ----------
    freq : np.ndarray
        Frequency vector.
    Pxx : np.ndarray
        Power spectral density.
    freq_bands : tuple, list, np.ndarray, optional
        Frequency bands to calculate rms values for. If None, the rms value of the whole spectrum is returned.
        If a tuple/list/array of length 2 is given, the rms value of the given frequency band is returned.
        If a 2D array is given, the rms values of each frequency band is returned.
        The default is None.
    exclude_zero_frequency : bool, optional
        If True, the DC component is discarded. The default is True.
        
    returns
    -------
    float | np.ndarray
        RMS value(s) of the power spectral density.
    """
    if len(freq) != len(Pxx):
        raise ValueError("Frequency vector and power spectral density must have the same length.")
    if len(freq) == 0 or len(Pxx) == 0:
        raise ValueError("Input arrays are empty.")
    if not isinstance(freq, np.ndarray):
        raise TypeError("Frequency vector must be a numpy array.")
    if not isinstance(Pxx, np.ndarray):
        raise TypeError("Power spectral density must be a numpy array.")
    
    if exclude_zero_frequency and freq[0] == 0.0: # always discard DC component
        freq = freq[1:]
        Pxx  = Pxx[1:]
    
    def rms_in_band(freq_band):
        mask = (freq >= freq_band[0]) & (freq < freq_band[1])
        return np.sqrt(np.sum(Pxx[mask]) * np.diff(freq[mask]).mean())

    if freq_bands is None:
        return np.sqrt(np.sum(Pxx) * np.diff(freq).mean())
        
    elif isinstance(freq_bands, (tuple, list, np.ndarray)):
        if np.ndim(freq_bands) == 1:
            if len(freq_bands) != 2:
                raise ValueError("Frequency bands must have length 2.")
            return rms_in_band(freq_bands)
        else:
            return np.array([rms_in_band(band) for band in freq_bands])
    else:
        raise ValueError("Frequency bands must be a tuple/list/array or a 2D array.")
