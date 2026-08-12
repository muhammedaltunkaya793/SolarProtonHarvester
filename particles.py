"""Relativistic charged-particle model."""

from dataclasses import dataclass
import numpy as np
from constants import C, PROTON_MASS, E_CHARGE, EV


@dataclass
class Proton:
    position: np.ndarray
    velocity: np.ndarray
    charge: float = E_CHARGE
    mass: float = PROTON_MASS

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

    @property
    def speed(self):
        return float(np.linalg.norm(self.velocity))

    @property
    def beta(self):
        return self.speed / C

    @property
    def gamma(self):
        beta2 = self.beta ** 2
        if beta2 >= 1.0:
            raise ValueError("Particle speed must be below c.")
        return 1.0 / np.sqrt(1.0 - beta2)

    @property
    def momentum(self):
        return self.gamma * self.mass * self.velocity

    @property
    def rest_energy(self):
        return self.mass * C**2

    @property
    def total_energy(self):
        return self.gamma * self.rest_energy

    @property
    def kinetic_energy(self):
        return (self.gamma - 1.0) * self.rest_energy

    def energy_momentum_check(self):
        lhs = self.total_energy**2
        rhs = (np.linalg.norm(self.momentum) * C)**2 + self.rest_energy**2
        return abs(lhs - rhs) / max(abs(lhs), 1e-300)

    def kinetic_energy_ev(self):
        return self.kinetic_energy / EV
