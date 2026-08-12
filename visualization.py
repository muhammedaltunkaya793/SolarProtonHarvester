"""Visualization tools for the Solar Proton Energy Harvester."""

import matplotlib.pyplot as plt
import numpy as np


C = 299_792_458.0
E_CHARGE = 1.602176634e-19


def plot_trajectory(result):
    positions = np.asarray(result["positions"])

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        x,
        y,
        z,
        linewidth=2.0,
        label="Proton trajectory"
    )

    ax.scatter(
        x[0],
        y[0],
        z[0],
        s=70,
        label="Initial position"
    )

    ax.scatter(
        x[-1],
        y[-1],
        z[-1],
        s=70,
        label="Final position"
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")

    ax.set_title(
        "Relativistic Proton Trajectory"
    )

    ax.legend()
    fig.tight_layout()


def plot_energy(result):
    time = np.asarray(result["time"])
    energy = np.asarray(result["energies"])

    kinetic_energy_ev = energy / E_CHARGE

    fig = plt.figure(figsize=(10, 6))

    plt.plot(
        time,
        kinetic_energy_ev,
        linewidth=2.0
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic Energy (eV)")
    plt.title("Relativistic Proton Kinetic Energy")

    plt.grid(True, alpha=0.25)
    plt.tight_layout()


def plot_speed(result):
    time = np.asarray(result["time"])
    velocities = np.asarray(result["velocities"])

    speed = np.linalg.norm(velocities, axis=1)
    normalized_speed = speed / C

    fig = plt.figure(figsize=(10, 6))

    plt.plot(
        time,
        normalized_speed,
        linewidth=2.0
    )

    plt.xlabel("Time (s)")
    plt.ylabel("v/c")
    plt.title("Relativistic Proton Speed")

    plt.grid(True, alpha=0.25)
    plt.tight_layout()


def plot_trajectory_xy(result):
    """2D projection of the proton trajectory onto the x-y plane."""

    positions = np.asarray(result["positions"])

    x = positions[:, 0]
    y = positions[:, 1]

    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        y,
        linewidth=2.0,
        label="Proton trajectory"
    )

    plt.scatter(
        x[0],
        y[0],
        s=70,
        label="Initial position"
    )

    plt.scatter(
        x[-1],
        y[-1],
        s=70,
        label="Final position"
    )

    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Proton Trajectory — x-y Projection")

    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()