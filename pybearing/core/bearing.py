import numpy as np
from dataclasses import dataclass, fields
from typing import Optional


@dataclass(frozen=True)
class Bearing:
    """
    Bearing dataclass.

    Attributes
    ----------
    name : str
        Name or identifier of the bearing.
    d : float
        Inner diameter in mm.
    D : float
        Outer diameter in mm.
    B : float
        Width in mm.
    D_b : Optional[float]
        Diameter of the rolling element (ball) in mm (Optional).
    D_m : Optional[float]
        Mean diameter in mm (Optional).
    theta : Optional[float]
        Contact angle in radians (Optional).
    N_b : Optional[int]
        Number of rolling elements (Optional).
    bearing_type : Optional[str]
        Type of the bearing (e.g., "Deep Groove Ball Bearing") (Optional).
    manufacturer : Optional[str]
        Manufacturer of the bearing (optional).
    """
    name: str                           # Name or identifier of the bearing
    d: float                            # Inner diameter in mm
    D: float                            # Outer diameter in mm
    B: float                            # Width in mm
    D_b: Optional[float] = None         # Diameter of the rolling element (ball) in mm (Optional)
    D_m: Optional[float] = None         # Mean diameter in mm (Optional)
    theta: Optional[float] = None       # Contact angle in radians (Optional)
    N_b: Optional[int] = None           # Number of rolling elements
    bearing_type: Optional[str] = None  # Type of the bearing (e.g., "Deep Groove Ball Bearing")
    manufacturer: Optional[str] = None  # Manufacturer of the bearing (optional)

    def __post_init__(self):
        # Validate internal geometry parameters if they are provided for future calculations of fault frequencies
        if self.D_b is not None and self.D_b <= 0:
            raise ValueError("Diameter of rolling element D_b must be positive")

        if self.D_m is not None and self.D_m <= 0:
            raise ValueError("Mean diameter D_m must be positive")

        if self.theta is not None and (self.theta < 0 or self.theta > np.pi/2):
            raise ValueError("Contact angle theta must be between 0 and pi/2 radians")

        if self.N_b is not None and self.N_b <= 2:
            raise ValueError("Number of rolling elements N_b must be greater than 2")

        object.__setattr__(self,
                           "internal_geometry_specified",
                           all(v is not None for v in [self.D_b, self.D_m, self.theta, self.N_b]))

    def __str__(self):
        return (
            f"Bearing {self.name} ({self.bearing_type}):\n"
            f"  Inner Diameter: {self.d} mm\n"
            f"  Outer Diameter: {self.D} mm\n"
            f"  Width: {self.B} mm\n"
            f"  Rolling element Diameter: {self.D_b or 'Not specified'} mm\n"
            f"  Mean Diameter: {self.D_m or 'Not specified'} mm\n"
            f"  Contact Angle: {self.theta or 'Not specified'} rad\n"
            f"  Number of Rolling Elements: {self.N_b or 'Not specified'}\n"
            f"  Manufacturer: {self.manufacturer or 'Not specified'}\n"
        )
    
    @classmethod
    def from_dict(cls, data:dict) -> "Bearing":
        valid_keys = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)
