"""Electromagnetic field models."""

from dataclasses import dataclass
import numpy as np


@dataclass
class UniformField:
    electric: np.ndarray
    magnetic: np.ndarray

    def __post_init__(self):
        self.electric = np.asarray(self.electric, dtype=float)
        self.magnetic = np.asarray(self.magnetic, dtype=float)

    def at(self, position, time=0.0):
        return self.electric.copy(), self.magnetic.copy()


@dataclass
class SolarWindField:
    """Ideal-MHD motional electric field E = -v_sw x B."""

    solar_wind_velocity: np.ndarray
    magnetic_field: np.ndarray

    def __post_init__(self):
        self.solar_wind_velocity = np.asarray(self.solar_wind_velocity, dtype=float)
        self.magnetic_field = np.asarray(self.magnetic_field, dtype=float)

    @property
    def electric_field(self):
        return -np.cross(self.solar_wind_velocity, self.magnetic_field)

    def at(self, position, time=0.0):
        return self.electric_field.copy(), self.magnetic_field.copy()
