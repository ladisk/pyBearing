from dataclasses import dataclass, fields
from typing import Optional


@dataclass
class FaultFrequencies:
    """
    A dataclass to represent the characteristic fault frequencies of a bearing.

    Attributes
    ----------
    f_cage : float
        Frequency of the cage (Hz)
    f_rolling_element_about_axis : float
        Frequency of the rolling element about its own axis (Hz)
    f_outer_ring : float
        Frequency of the outer ring (Hz)
    f_rolling_element : float
        Frequency of the rolling element (Hz)
    f_inner_ring : float
        Frequency of the inner ring (Hz)
    f_of_rotation : Optional[float]
        Frequency of rotation (Hz, optional, for reference)
    """
    f_cage: float                           # Frequency of the cage
    f_rolling_element_about_axis: float     # Frequency of the rolling element about its own axis
    f_outer_ring: float                     # Frequency of the outer ring
    f_rolling_element: float                # Frequency of the rolling element
    f_inner_ring: float                     # Frequency of the inner ring
    f_of_rotation: Optional[float] = None   # Frequency of rotation (optional, for reference)

    def __post_init__(self):
        if self.f_cage <= 0:
            raise ValueError("Cage frequency f_cage must be positive")
        if self.f_rolling_element_about_axis < 0:
            raise ValueError("Rolling element about axis frequency f_rolling_element_about_axis must be positive")
        if self.f_outer_ring < 0:
            raise ValueError("Outer ring frequency f_outer_ring must be positive")
        if self.f_rolling_element < 0:
            raise ValueError("Rolling element frequency f_rolling_element must be positive")
        if self.f_inner_ring < 0:
            raise ValueError("Inner ring frequency f_inner_ring must be positive")
        if self.f_of_rotation is not None and self.f_of_rotation < 0:
            raise ValueError("Frequency of rotation f_of_rotation must be positive if provided")

    def __str__(self):
        return (f"Fault Frequencies at {self.f_of_rotation} Hz:\n"
                f"  Cage: {self.f_cage} Hz\n"
                f"  Rolling Element About Axis: {self.f_rolling_element_about_axis} Hz\n"
                f"  Outer Ring: {self.f_outer_ring} Hz\n"
                f"  Rolling Element: {self.f_rolling_element} Hz\n"
                f"  Inner Ring: {self.f_inner_ring} Hz")
    
    def __mul__(self, factor: float):
        if not isinstance(factor, (int, float)):
            return NotImplemented

        return FaultFrequencies(
            f_cage=round(self.f_cage * factor, 4),
            f_rolling_element_about_axis=round(self.f_rolling_element_about_axis * factor, 4),
            f_outer_ring=round(self.f_outer_ring * factor, 4),
            f_rolling_element=round(self.f_rolling_element * factor, 4),
            f_inner_ring=round(self.f_inner_ring * factor, 4),
            f_of_rotation=(
                self.f_of_rotation * factor
                if self.f_of_rotation is not None
                else None
            )
        )

    __rmul__ = __mul__
    
    def fault_frequencies(self):
        return {
            "f_cage": self.f_cage,
            "f_rolling_element_about_axis": self.f_rolling_element_about_axis,
            "f_outer_ring": self.f_outer_ring,
            "f_rolling_element": self.f_rolling_element,
            "f_inner_ring": self.f_inner_ring
        }

    @classmethod
    def from_dict_database(cls, data:dict) -> "FaultFrequencies":
        mapping = {
            "f_rot": "f_of_rotation",
            "f_c": "f_cage",
            "f_re_ax": "f_rolling_element_about_axis",
            "f_o": "f_outer_ring",
            "f_re": "f_rolling_element",
            "f_i": "f_inner_ring"
        }
        data = {mapping.get(key, key): value for key, value in data.items()}
        valid_keys = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)
