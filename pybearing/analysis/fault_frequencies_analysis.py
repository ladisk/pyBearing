import numpy as np

from ..core.bearing import Bearing
from ..core.fault_frequencies import FaultFrequencies
from ..data.bearing_database import BearingDatabase


# Sources for equations:
# Randall, R.B.. (2004). State of the Art in Monitoring Rotating Machinery - Part 1. Sound & vibration. 38. 14-21+13.
# J. Slavič, A Brković and M. Boltežar, Typical bearing-fault rating using force measurements: application to real data. Journal of Vibration and Control, December 2011 vol. 17 no. 14 2164-2174.. DOI: 10.1177/1077546311399949

def eq_f_cage(bearing:Bearing, frequency_of_rotation:float = 1.0, inner_rotating:bool = True) -> float:
    """
    Calculate the cage frequency of a bearing.

    parameters
    ----------
    bearing : Bearing
        The bearing for which to calculate the cage frequency.
    frequency_of_rotation : float
        Frequency of rotation in Hz.
    inner_rotating : bool
        Whether the inner ring is rotating (default is True).

    returns
    -------
    float
        The cage frequency in Hz.
    """
    sign = -1 if inner_rotating else 1
    f_c = round(0.5 * frequency_of_rotation * (1 + sign*(bearing.D_b/bearing.D_m)*np.cos(bearing.theta)), 4)
    return float(f_c)

def eq_f_rolling_element_about_axis(bearing:Bearing, frequency_of_rotation:float = 1.0) -> float:
    """
    Calculate the frequency of the rolling element about its own axis.

    parameters
    ----------
    bearing : Bearing
        The bearing for which to calculate the frequency.
    frequency_of_rotation : float
        Frequency of rotation in Hz.

    returns
    -------
    float
        The frequency of the rolling element about its own axis in Hz.
    """
    f_re_ax = round(0.5 * (bearing.D_m/bearing.D_b) * frequency_of_rotation * (1 - (bearing.D_b**2/bearing.D_m**2)*np.cos(bearing.theta)**2), 4)
    return float(f_re_ax)

def eq_f_outer_ring(bearing:Bearing, frequency_of_rotation:float = 1.0) -> float:
    """
    Calculate the frequency of the outer ring.

    parameters
    ----------
    bearing : Bearing
        The bearing for which to calculate the frequency.
    frequency_of_rotation : float
        Frequency of rotation in Hz.

    returns
    -------
    float
        The frequency of the outer ring in Hz.
    """
    f_o = round(0.5 * bearing.N_b * frequency_of_rotation * (1 - (bearing.D_b/bearing.D_m)*np.cos(bearing.theta)), 4)
    return float(f_o)

def eq_f_rolling_element(bearing:Bearing, frequency_of_rotation:float = 1.0) -> float:
    """
    Calculate the frequency of the rolling element.

    parameters
    ----------
    bearing : Bearing
        The bearing for which to calculate the frequency.
    frequency_of_rotation : float
        Frequency of rotation in Hz.

    returns
    -------
    float
        The frequency of the rolling element in Hz.
    """
    f_re_ax = eq_f_rolling_element_about_axis(bearing, frequency_of_rotation)
    f_re = round(2 * f_re_ax, 4)
    return float(f_re)

def eq_f_inner_ring(bearing:Bearing, frequency_of_rotation:float = 1.0) -> float:
    """
    Calculate the frequency of the inner ring.

    parameters
    ----------
    bearing : Bearing
        The bearing for which to calculate the frequency.
    frequency_of_rotation : float
        Frequency of rotation in Hz.

    returns
    -------
    float
        The frequency of the inner ring in Hz.
    """
    f_i = round(0.5 * bearing.N_b * frequency_of_rotation * (1 + (bearing.D_b/bearing.D_m)*np.cos(bearing.theta)), 4)
    return float(f_i)

def calculate_from_geometry(bearing:Bearing, rpm:float, inner_rotating:bool = True) -> FaultFrequencies:
    frequency_of_rotation = round(rpm / 60.0, 4)
    f_cage = eq_f_cage(bearing, frequency_of_rotation, inner_rotating)
    f_rolling_element_about_axis = eq_f_rolling_element_about_axis(bearing, frequency_of_rotation)
    f_outer_ring = eq_f_outer_ring(bearing, frequency_of_rotation)
    f_rolling_element = eq_f_rolling_element(bearing, frequency_of_rotation)
    f_inner_ring = eq_f_inner_ring(bearing, frequency_of_rotation)

    return FaultFrequencies(
        f_cage=f_cage,
        f_rolling_element_about_axis=f_rolling_element_about_axis,
        f_outer_ring=f_outer_ring,
        f_rolling_element=f_rolling_element,
        f_inner_ring=f_inner_ring,
        f_of_rotation=frequency_of_rotation
    )

def calculate_from_database(bearing:Bearing, rpm:float, database:BearingDatabase, inner_rotating:bool = True) -> FaultFrequencies:
    frequency_of_rotation = round(rpm / 60.0, 4)
    found_characteristic_frequencies = database.find_characteristic_frequencies_by_bearing_name(bearing.name)

    for i, freq in enumerate(found_characteristic_frequencies):
        if inner_rotating is not True:
            freq.f_cage = round(1 - freq.f_cage, 4)
        found_characteristic_frequencies[i] = freq * frequency_of_rotation
    
    if len(found_characteristic_frequencies) == 0:
        raise ValueError(f"No characteristic frequencies found for bearing '{bearing.name}' in the database.")
    elif len(found_characteristic_frequencies) == 1:
        return found_characteristic_frequencies[0]
    else:
        return found_characteristic_frequencies

def get_fault_frequencies(bearing:Bearing, rpm:float, inner_rotating:bool = True, method:str = "geometry", database:BearingDatabase = None) -> FaultFrequencies:
    if method == "geometry":
        return calculate_from_geometry(bearing, rpm, inner_rotating)

    elif method == "database":
        # If database is not provided, use the default database
        if database is None:
            database = BearingDatabase()
        return calculate_from_database(bearing, rpm, database, inner_rotating)

    else:
        raise ValueError("Unknown method")
