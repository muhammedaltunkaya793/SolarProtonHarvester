"""Energy-harvesting accounting model."""

from dataclasses import dataclass
from constants import E_CHARGE, EV


@dataclass
class Harvester:
    area: float = 1.0                 # m^2
    conversion_efficiency: float = 0.20
    collection_efficiency: float = 0.50

    def intercepted_power(self, solar_wind):
        return solar_wind.kinetic_energy_flux() * self.area * self.collection_efficiency

    def usable_power(self, solar_wind):
        return self.intercepted_power(solar_wind) * self.conversion_efficiency

    def particle_event_energy(self, kinetic_energy_joule):
        return kinetic_energy_joule * self.collection_efficiency * self.conversion_efficiency

    def summary(self, solar_wind):
        intercepted = self.intercepted_power(solar_wind)
        usable = self.usable_power(solar_wind)
        return {
            "kinetic_energy_flux_W_m2": solar_wind.kinetic_energy_flux(),
            "intercepted_power_W": intercepted,
            "usable_power_W": usable,
            "effective_efficiency": self.collection_efficiency * self.conversion_efficiency,
        }
