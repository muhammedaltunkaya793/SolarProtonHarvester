"""
Research analysis for the Solar Proton Energy Harvester.

This module performs:
1. timestep-convergence analysis;
2. solar-wind parameter sweeps;
3. reproducible CSV output;
4. publication-style diagnostic plots.

Run:
    python3 research_analysis.py
"""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from constants import EV
from particles import Proton
from solar_wind import SolarWind
from fields import SolarWindField
from integrator import simulate
from harvester import Harvester


RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def build_proton():
    """Create the standard relativistic test proton."""
    from constants import C

    return Proton(
        position=np.zeros(3),
        velocity=np.array([0.9 * C, 0.0, 0.0]),
    )


def build_field(wind):
    """Create the standard ideal-MHD electromagnetic environment."""
    return SolarWindField(
        solar_wind_velocity=np.array(
            [wind.bulk_speed, 0.0, 0.0]
        ),
        magnetic_field=np.array(
            [0.0, 5.0e-9, 0.0]
        ),
    )


def timestep_convergence():
    """Measure final kinetic energy versus timestep."""

    timesteps = np.array([
        5.0e-11,
        1.0e-10,
        2.0e-10,
        5.0e-10,
    ])

    wind = SolarWind(
        density=5.0e6,
        bulk_speed=400_000.0,
        temperature=1.0e5,
    )

    field = build_field(wind)

    rows = []

    for dt in timesteps:
        proton = build_proton()

        result = simulate(
            proton=proton,
            field=field,
            dt=dt,
            steps=int(2.0e-7 / dt),
        )

        initial_ke = result["energies"][0]
        final_ke = result["energies"][-1]

        relative_change = (
            (final_ke - initial_ke) / initial_ke
        )

        rows.append(
            {
                "dt_s": dt,
                "initial_energy_eV": initial_ke / EV,
                "final_energy_eV": final_ke / EV,
                "relative_energy_change": relative_change,
            }
        )

    output = RESULTS_DIR / "timestep_convergence.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    plt.figure()
    plt.loglog(
        timesteps,
        [
            abs(row["relative_energy_change"])
            for row in rows
        ],
        marker="o",
    )

    plt.xlabel("Time step (s)")
    plt.ylabel("|Relative kinetic-energy change|")
    plt.title("Timestep Convergence")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "timestep_convergence.png",
        dpi=300,
    )

    plt.close()

    return rows


def solar_wind_parameter_sweep():
    """Explore usable power over solar-wind conditions."""

    densities = np.array([
        1.0e6,
        2.5e6,
        5.0e6,
        1.0e7,
        2.0e7,
    ])

    speeds = np.array([
        2.0e5,
        3.0e5,
        4.0e5,
        5.0e5,
        6.0e5,
    ])

    harvester = Harvester(
        area=1.0,
        conversion_efficiency=0.20,
        collection_efficiency=0.50,
    )

    rows = []

    for density in densities:
        for speed in speeds:

            wind = SolarWind(
                density=density,
                bulk_speed=speed,
                temperature=1.0e5,
            )

            summary = harvester.summary(wind)

            rows.append(
                {
                    "density_m^-3": density,
                    "bulk_speed_m_s": speed,
                    "kinetic_energy_flux_W_m^-2":
                        wind.kinetic_energy_flux(),
                    "usable_power_W":
                        summary["usable_power_W"],
                }
            )

    output = RESULTS_DIR / "solar_wind_parameter_sweep.csv"

    with output.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    matrix = np.zeros(
        (len(densities), len(speeds))
    )

    for row in rows:
        i = np.where(
            densities == row["density_m^-3"]
        )[0][0]

        j = np.where(
            speeds == row["bulk_speed_m_s"]
        )[0][0]

        matrix[i, j] = row["usable_power_W"]

    plt.figure()

    image = plt.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        extent=[
            speeds[0],
            speeds[-1],
            densities[0],
            densities[-1],
        ],
    )

    plt.colorbar(image, label="Usable power (W)")
    plt.xlabel("Solar-wind bulk speed (m/s)")
    plt.ylabel("Number density (m⁻³)")
    plt.title("Solar-Wind Parameter Sweep")
    plt.tight_layout()

    plt.savefig(
        RESULTS_DIR / "solar_wind_parameter_sweep.png",
        dpi=300,
    )

    plt.close()

    return rows


def main():
    print("=" * 60)
    print("SOLAR PROTON ENERGY HARVESTER")
    print("Research Analysis")
    print("=" * 60)

    print("\nRunning timestep-convergence analysis...")
    convergence = timestep_convergence()

    print(
        f"Generated {len(convergence)} convergence cases."
    )

    print("\nRunning solar-wind parameter sweep...")
    sweep = solar_wind_parameter_sweep()

    print(
        f"Generated {len(sweep)} parameter combinations."
    )

    print("\nResults written to:")
    print(f"  {RESULTS_DIR / 'timestep_convergence.csv'}")
    print(f"  {RESULTS_DIR / 'timestep_convergence.png'}")
    print(
        f"  {RESULTS_DIR / 'solar_wind_parameter_sweep.csv'}"
    )
    print(
        f"  {RESULTS_DIR / 'solar_wind_parameter_sweep.png'}"
    )

    print("\nAnalysis completed successfully.")


if __name__ == "__main__":
    main()