"""Solar Proton Energy Harvester - research demonstrator.

Run:
    python3 main.py

The program combines:
1. relativistic proton kinematics;
2. solar-wind bulk properties;
3. an ideal-MHD motional electric field;
4. relativistic charged-particle integration;
5. energy-harvesting accounting;
6. numerical consistency diagnostics.
"""

import numpy as np

from constants import C, PROTON_MASS, E_CHARGE, EV, MEV
from particles import Proton
from solar_wind import SolarWind
from fields import SolarWindField
from integrator import simulate
from harvester import Harvester
from visualization import plot_trajectory, plot_energy, plot_speed


def print_header():
    print("=" * 62)
    print("=== SOLAR PROTON ENERGY HARVESTER ===")
    print("Relativistic charged-particle research demonstrator")
    print("=" * 62)


def main():
    print_header()

    # --- Solar-wind environment ---
    wind = SolarWind(
        density=5.0e6,
        bulk_speed=400_000.0,
        temperature=1.0e5,
    )

    # A representative magnetic-field scale near Earth.
    # Orientation is chosen perpendicular to the solar-wind flow
    # so the ideal-MHD motional electric field is non-zero.
    field = SolarWindField(
        solar_wind_velocity=np.array([wind.bulk_speed, 0.0, 0.0]),
        magnetic_field=np.array([0.0, 5.0e-9, 0.0]),
    )

    # --- Relativistic proton test particle ---
    # 0.9c is deliberately chosen as a relativistic stress test.
    v = np.array([0.9 * C, 0.0, 0.0])
    proton = Proton(position=np.zeros(3), velocity=v)

    print("\n--- RELATIVISTIC PROTON ---")
    print(f"Speed:              {proton.speed:.9e} m/s")
    print(f"v/c:                {proton.beta:.9f}")
    print(f"Lorentz gamma:      {proton.gamma:.9f}")
    print(f"Rest energy:        {proton.rest_energy:.9e} J")
    print(f"Total energy:       {proton.total_energy:.9e} J")
    print(f"Kinetic energy:     {proton.kinetic_energy:.9e} J")
    print(f"Kinetic energy:     {proton.kinetic_energy / EV:.6e} eV")
    print(f"Kinetic energy:     {proton.kinetic_energy / MEV:.6f} MeV")
    print(f"Momentum magnitude: {np.linalg.norm(proton.momentum):.9e} kg*m/s")

    error = proton.energy_momentum_check()
    print("\n=== RELATIVISTIC CONSISTENCY CHECK ===")
    print("E² = (pc)² + (mc²)²")
    print("PASS" if error < 1e-12 else "WARNING")
    print(f"Relative numerical error: {error:.3e}")

    # --- Solar wind energy budget ---
    print("\n--- SOLAR-WIND ENVIRONMENT ---")
    print(f"Number density:     {wind.density:.3e} m^-3")
    print(f"Bulk speed:         {wind.bulk_speed:.3e} m/s")
    print(f"Temperature:        {wind.temperature:.3e} K")
    print(f"1D thermal speed:   {wind.thermal_speed_1d:.3e} m/s")
    print(f"Particle flux:      {wind.particle_flux():.3e} m^-2 s^-1")
    print(f"Kinetic-energy flux: {wind.kinetic_energy_flux():.6e} W/m^2")
    print(f"Motional |E|:       {np.linalg.norm(field.electric_field):.6e} V/m")
    print(f"|B|:                {np.linalg.norm(field.magnetic_field):.6e} T")

    # --- Harvester budget ---
    harvester = Harvester(
        area=1.0,
        conversion_efficiency=0.20,
        collection_efficiency=0.50,
    )
    h = harvester.summary(wind)

    print("\n--- HARVESTER ENERGY BUDGET ---")
    print(f"Collector area:     {harvester.area:.3f} m^2")
    print(f"Collection eff.:    {harvester.collection_efficiency:.1%}")
    print(f"Conversion eff.:    {harvester.conversion_efficiency:.1%}")
    print(f"Intercepted power:  {h['intercepted_power_W']:.6e} W")
    print(f"Usable power:       {h['usable_power_W']:.6e} W")

    # --- Relativistic trajectory ---
    # The time step is chosen for a short diagnostic run.
    # For a production study, perform timestep-convergence tests.
    dt = 1.0e-10
    steps = 2000

    result = simulate(
        proton=proton,
        field=field,
        dt=dt,
        steps=steps,
    )

    energy_change = result["energies"][-1] - result["energies"][0]

    print("\n--- PARTICLE DYNAMICS ---")
    print(f"Simulation time:    {result['time'][-1]:.3e} s")
    print(f"Time step:          {dt:.3e} s")
    print(f"Steps:              {steps}")
    print(f"Initial KE:         {result['energies'][0] / EV:.6e} eV")
    print(f"Final KE:           {result['energies'][-1] / EV:.6e} eV")
    print(f"Delta KE:           {energy_change / EV:.6e} eV")

    # Visual diagnostics.
    plot_trajectory(result)
    plot_energy(result)
    plot_speed(result)

    print("\nPlots generated. Close the plot windows to finish.")
    import matplotlib.pyplot as plt
    plt.show()


if __name__ == "__main__":
    main()
