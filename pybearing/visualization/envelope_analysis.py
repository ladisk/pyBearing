import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from ..core.fault_frequencies import FaultFrequencies
from ..analysis.filter_search_results import FilterSearchResults


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

    # Check if the provided rotation frequency matches the one from the fault frequencies.
    # If not, transform the fault frequencies to match the provided rotation frequency.
    if f_of_rotation != f_of_rotation_from_fault_frequencies:
        print("Warning: The provided rotation frequency does not match the one from the fault frequencies. Transforming the fault frequencies to match the provided rotation frequency.")
        vlines_x_value = vlines_x_value * (f_of_rotation / f_of_rotation_from_fault_frequencies)
        upper_x_limit = 1.1 * np.max(vlines_x_value)
    
    # normalise the frequency axis by the rotation frequency if normalise is True
    if normalise:
        freq = freq / f_of_rotation
        vlines_x_value = vlines_x_value / f_of_rotation
        upper_x_limit = upper_x_limit / f_of_rotation
        
    plt.figure()
    plt.plot(freq, np.abs(ampl))
    plt.vlines(vlines_x_value, ymin=0, ymax=np.max(np.abs(ampl)), colors="red", linestyles="dashed", label="Fault frequencies")
    plt.xlabel("Frequency [Hz]") if not normalise else plt.xlabel("Frequency/frequency_of_rotation [/]")
    plt.ylabel("Amplitude")
    plt.xlim(0, upper_x_limit)
    plt.grid()
    plt.legend()

def plot_filter_search_for_envelope_extraction(
        filter_search_results:FilterSearchResults,
        plot_harmonics_score:bool = True,
        plot_kurtosis_score:bool = False
    ):
    names_for_plot = []
    scores_for_plot = []
    if plot_harmonics_score and filter_search_results.harmonics_score is not None:
        fault_types = list(filter_search_results.harmonics_score.__annotations__.keys())
        names_for_plot += fault_types
        scores_for_plot.extend(filter_search_results.harmonics_score.__getattribute__(key) for key in fault_types)
    elif plot_harmonics_score and filter_search_results.harmonics_score is None:
        print("Warning: plot_harmonics_score is True but harmonics_score is missing.")
    
    if plot_kurtosis_score and filter_search_results.kurtosis_score is not None:
        names_for_plot.append("kurtosis")
        scores_for_plot.append(filter_search_results.kurtosis_score)
    elif plot_kurtosis_score and filter_search_results.kurtosis_score is None:
        print("Warning: plot_kurtosis_score is True but kurtosis_score is missing.")

    for name, plot_score in zip(names_for_plot, scores_for_plot):
        levels = sorted(set(filter_search_results.level))

        # map actual level -> row number
        level_to_row = {lvl: i for i, lvl in enumerate(levels)}

        fig, ax = plt.subplots(figsize=(12, 6))

        norm = Normalize(vmin=min(plot_score), vmax=max(plot_score))
        cmap = plt.cm.viridis

        for f_low, f_high, level, score in zip(
            filter_search_results.f_low,
            filter_search_results.f_high,
            filter_search_results.level,
            plot_score
        ):

            row = level_to_row[level]

            rect = Rectangle(
                (f_low, row),
                f_high - f_low,
                1.0,
                facecolor=cmap(norm(score)),
                edgecolor="none"
            )

            ax.add_patch(rect)

        ax.relim()
        ax.autoscale_view()
        ax.set_ylim(len(levels), 0)
        ax.set_xlim(min(filter_search_results.f_low), max(filter_search_results.f_high))
        ax.set_yticks(np.arange(len(levels)) + 0.5)
        ax.set_yticklabels([f"{lvl:.1f}" for lvl in levels])

        ax.set_ylabel("Level")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_title(f"{str(name).replace('_', ' ').title()}")

        sm = ScalarMappable(norm=norm, cmap=cmap) 
        plt.colorbar(sm, ax=ax, label="Score")
        plt.show()
