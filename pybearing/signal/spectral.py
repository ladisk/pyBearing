import numpy as np
from scipy.signal import csd


def fft(x:np.ndarray, fs:int, ret_freq:bool=False)->np.ndarray|tuple[np.ndarray, np.ndarray]:
    """
    Calculates the FFT of a signal.
    
    Fourier transform of a signal ``x(t)`` is normed so that the ``|X(f)|`` 
    gives the amplitude of the signal at frequency f.
    
    parameters
    ----------
    x : np.ndarray
        Input time signal.
    fs : int
        Sampling frequency in Hz.
    ret_freq : bool, optional
        If True, the frequency vector is returned as well. The default is False.
    
    returns
    -------
    np.ndarray | tuple[np.ndarray, np.ndarray]
        array with complex values representing FFT of the input signal.
        If ret_freq is True, a tuple is returned with the frequency vector as well (freq, X).
    """
    if len(x) == 0:
        raise ValueError("Input array is empty.")
    
    X = np.fft.rfft(x)
    X[1:] *= 2/len(x) # normalize
    X[0]  *= 1/len(x) # normalize
    
    if ret_freq:
        freq = np.fft.rfftfreq(len(x), 1/fs)
        return freq, X
    else:
        return X

def power_spectral_density(x:np.ndarray, fs:int, nperseg:int=None, noverlap:int=None, window:np.ndarray|str="hann") -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the power spectral density of a signal.

    parameters
    ----------
    x : np.ndarray
        The input signal.
    fs : int
        The sampling frequency of the signal.
    nperseg : int, optional
        The length of each segment. Default is None.
    noverlap : int, optional
        The number of points to overlap between segments. If None, noverlap = nperseg // 2.
    window : np.ndarray or str, optional
        The window to apply to each segment. Default is "hann".

    returns
    -------
    freq : np.ndarray
        The frequencies corresponding to the power spectral density values.
    Pxx : np.ndarray
        The power spectral density of the signal.
    """
    if len(x) == 0:
        raise ValueError("Input array is empty.")

    freq, Pxx = csd(x, x, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window, scaling="density")
    Pxx = np.abs(Pxx)
    return freq, Pxx
