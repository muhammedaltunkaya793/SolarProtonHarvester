import numpy as np

from constants import C, PROTON_MASS, E_CHARGE
from particles import Proton


def test_light_speed_constant():
    assert C > 2.9e8
    assert C < 3.1e8


def test_proton_rest_energy():
    proton = Proton(
        position=np.zeros(3),
        velocity=np.array([0.0, 0.0, 0.0])
    )

    expected = PROTON_MASS * C**2

    assert np.isclose(
        proton.rest_energy,
        expected,
        rtol=1e-12
    )


def test_rest_particle_has_zero_kinetic_energy():
    proton = Proton(
        position=np.zeros(3),
        velocity=np.zeros(3)
    )

    assert np.isclose(
        proton.kinetic_energy,
        0.0,
        atol=1e-30
    )


def test_relativistic_energy_momentum_relation():
    proton = Proton(
        position=np.zeros(3),
        velocity=np.array([0.9 * C, 0.0, 0.0])
    )

    relative_error = proton.energy_momentum_check()

    assert relative_error < 1e-12


def test_proton_charge():
    proton = Proton(
        position=np.zeros(3),
        velocity=np.zeros(3)
    )

    assert np.isclose(
        proton.charge,
        E_CHARGE,
        rtol=1e-12
    )


def test_gamma_greater_than_one_for_moving_proton():
    proton = Proton(
        position=np.zeros(3),
        velocity=np.array([0.9 * C, 0.0, 0.0])
    )

    assert proton.gamma > 1.0


def test_speed_remains_below_light_speed():
    proton = Proton(
        position=np.zeros(3),
        velocity=np.array([0.9 * C, 0.0, 0.0])
    )

    assert proton.speed < C