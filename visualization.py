"""Plots for trajectory and energy diagnostics."""

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(result):
    p = result["positions"]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(p[:, 0], p[:, 1], p[:, 2])
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title("Relativistic proton trajectory")
    fig.tight_layout()


def plot_energy(result):
    t = result["time"]
    e = result["energies"]
    plt.figure()
    plt.plot(t, e / 1.602176634e-19)
    plt.xlabel("Time (s)")
    plt.ylabel("Kinetic energy (eV)")
    plt.title("Proton kinetic-energy history")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()


def plot_speed(result):
    v = np.linalg.norm(result["velocities"], axis=1)
    plt.figure()
    plt.plot(result["time"], v / 299_792_458.0)
    plt.xlabel("Time (s)")
    plt.ylabel("v/c")
    plt.title("Relativistic speed history")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
