import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class PerFaultResults:
    """
    Per-fault-type results dataclass. Meant to be used where some result of analysis is calculated
    for each fault type separately (cage, rolling element, outer ring, rolling element, inner ring).
    
    Attributes
    ----------
    cage : np.ndarray
    rolling_element_about_axis : np.ndarray
    outer_ring : np.ndarray
    rolling_element : np.ndarray
    inner_ring : np.ndarray
    """
    cage: np.ndarray
    rolling_element_about_axis: np.ndarray
    outer_ring: np.ndarray
    rolling_element: np.ndarray
    inner_ring: np.ndarray

    def __post_init__(self):
        arrays = {
            "cage": self.cage,
            "rolling_element_about_axis": self.rolling_element_about_axis,
            "outer_ring": self.outer_ring,
            "rolling_element": self.rolling_element,
            "inner_ring": self.inner_ring,
        }

        shapes = {
            name: array.shape for name, array in arrays.items()
        }
        
        if len(set(shapes.values())) > 1:
            raise ValueError(
                f"All arrays must have the same shape. "
                f"Got: {shapes}"
            )

@dataclass(frozen=True)
class FilterSearchResults:
    """
    Stores results of the filter_search_for_envelope_extraction method. This dataclass is then
    used in the visualisation of the results and for further automatic selection of the best
    bandpass filters.

    Attributes
    ----------
    level: np.ndarray
        Determines the number of windows to be used in the envelope extraction.
    f_low: np.ndarray
        Lower cutoff frequencies of the bandpass filters used in the envelope extraction.
    f_high: np.ndarray
        Upper cutoff frequencies of the bandpass filters used in the envelope extraction.
    harmonics_score: PerFaultResults | None
        Score of the harmonics for each fault frequency.
    kurtosis_score: np.ndarray | None
        Score of the kurtosis of the envelope signal.
    """
    level: np.ndarray
    f_low: np.ndarray
    f_high: np.ndarray

    harmonics_score: PerFaultResults | None = None
    kurtosis_score: np.ndarray | None = None

    def __post_init__(self):
        arrays = {
            "level": self.level,
            "f_low": self.f_low,
            "f_high": self.f_high,
        }

        if self.harmonics_score is not None:
            arrays.update({
                "harmonics_score.cage": self.harmonics_score.cage,
                "harmonics_score.rolling_element_about_axis":
                    self.harmonics_score.rolling_element_about_axis,
                "harmonics_score.outer_ring": self.harmonics_score.outer_ring,
                "harmonics_score.rolling_element":
                    self.harmonics_score.rolling_element,
                "harmonics_score.inner_ring":
                    self.harmonics_score.inner_ring,
            })

        if self.kurtosis_score is not None:
            arrays["kurtosis_score"] = self.kurtosis_score

        shapes = {
            name: array.shape for name, array in arrays.items()
        }

        if len(set(shapes.values())) > 1:
            raise ValueError(
                f"All result arrays must have the same shape. "
                f"Got: {shapes}"
            )
