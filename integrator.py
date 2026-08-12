"""Relativistic charged-particle integrator.

Uses a relativistic Boris-style pusher. The state variable is momentum p,
which makes the relativistic update substantially cleaner than updating v
directly with a Newtonian acceleration formula.
"""

import numpy as np
from constants import C


def gamma_from_p(p, mass):
    p2 = float(np.dot(p, p))
    return np.sqrt(1.0 + p2 / (mass * C)**2)


def velocity_from_p(p, mass):
    gamma = gamma_from_p(p, mass)
    return p / (gamma * mass)


def relativistic_boris_step(position, momentum, charge, mass, field, dt, time):
    E, B = field.at(position, time)

    # Half electric impulse.
    p_minus = momentum + charge * E * (dt / 2.0)

    # Magnetic rotation.
    gamma_minus = gamma_from_p(p_minus, mass)
    t = charge * B * (dt / (2.0 * gamma_minus * mass))

    t2 = float(np.dot(t, t))
    s = 2.0 * t / (1.0 + t2)

    p_prime = p_minus + np.cross(p_minus, t)
    p_plus = p_minus + np.cross(p_prime, s)

    # Second half electric impulse.
    p_new = p_plus + charge * E * (dt / 2.0)

    v_new = velocity_from_p(p_new, mass)
    x_new = position + v_new * dt
    return x_new, p_new


def simulate(proton, field, dt, steps):
    positions = np.empty((steps + 1, 3))
    velocities = np.empty((steps + 1, 3))
    energies = np.empty(steps + 1)

    x = proton.position.copy()
    p = proton.momentum.copy()

    positions[0] = x
    velocities[0] = velocity_from_p(p, proton.mass)
    energies[0] = (gamma_from_p(p, proton.mass) - 1.0) * proton.mass * C**2

    for i in range(steps):
        x, p = relativistic_boris_step(
            x, p, proton.charge, proton.mass, field, dt, i * dt
        )
        positions[i + 1] = x
        velocities[i + 1] = velocity_from_p(p, proton.mass)
        energies[i + 1] = (gamma_from_p(p, proton.mass) - 1.0) * proton.mass * C**2

    return {
        "time": np.arange(steps + 1) * dt,
        "positions": positions,
        "velocities": velocities,
        "energies": energies,
    }
