"""Simple solar-wind population model.

The model is intentionally explicit about its assumptions:
- proton population is represented by a drifting thermal distribution;
- the distribution is not a full heliospheric plasma/MHD simulation;
- parameters can be changed for sensitivity studies.
"""

from dataclasses import dataclass
import numpy as np
from constants import PROTON_MASS, KB


@dataclass
class SolarWind:
    density: float = 5.0e6          # m^-3, representative near-Earth value
    bulk_speed: float = 400_000.0   # m/s
    temperature: float = 1.0e5     # K

    @property
    def thermal_speed_1d(self):
        return np.sqrt(KB * self.temperature / PROTON_MASS)

    def sample_protons(self, n=1000, seed=42):
        rng = np.random.default_rng(seed)
        thermal = rng.normal(0.0, self.thermal_speed_1d, size=(n, 3))
        thermal[:, 0] += self.bulk_speed
        return thermal

    def kinetic_energy_flux(self):
        # Bulk-flow kinetic-energy flux: 1/2 rho v^3.
        rho = self.density * PROTON_MASS
        return 0.5 * rho * self.bulk_speed**3

    def particle_flux(self):
        return self.density * self.bulk_speed
