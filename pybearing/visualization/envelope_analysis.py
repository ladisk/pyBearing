import matplotlib.pyplot as plt
import numpy as np

from ..core.fault_frequencies import FaultFrequencies


def plot_envelope(x:np.ndarray, envelope:np.ndarray, fs:int):
    time = np.arange(len(x)) / fs
    plt.figure()
    plt.plot(time, x, label="Measured signal")
    plt.plot(time, envelope, label="Envelope")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()

def plot_envelope_spectrum(freq:np.ndarray, ampl:np.ndarray, f_of_rotation:float, fault_frequencies:FaultFrequencies, normalise:bool=False):
    vlines_x_value = np.array(list(fault_frequencies.fault_frequencies().values()))
    upper_x_limit = 1.1 * np.max(vlines_x_value)

    f_of_rotation_from_fault_frequencies = fault_frequencies.f_of_rotation

    if f_of_rotation != f_of_rotation_from_fault_frequencies:
        print("Warning: The provided rotation frequency does not match the one from the fault frequencies. Transforming the fault frequencies to match the provided rotation frequency.")
        vlines_x_value = vlines_x_value * (f_of_rotation / f_of_rotation_from_fault_frequencies)
    
    if normalise:
        freq = freq / f_of_rotation
        vlines_x_value = vlines_x_value / f_of_rotation
        upper_x_limit = upper_x_limit / f_of_rotation
        
    plt.figure()
    plt.plot(freq, np.abs(ampl))
    plt.vlines(vlines_x_value, ymin=0, ymax=np.max(np.abs(ampl)), colors="red", linestyles="dashed", label="Fault frequencies")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.xlim(0, upper_x_limit)
    plt.grid()
    plt.legend()