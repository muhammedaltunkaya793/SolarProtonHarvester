# Solar Proton Energy Harvester

A computational-physics research demonstrator for investigating the energy
budget and relativistic dynamics of charged protons in a solar-wind
environment.

## Scientific question

> Can kinetic energy carried by solar-wind protons be converted into useful
> electrical energy, and what limits the conversion?

## Model layers

1. **Relativistic kinematics**
   - Lorentz factor
   - total, rest and kinetic energy
   - relativistic momentum
   - energy-momentum consistency check

2. **Solar-wind model**
   - number density
   - bulk velocity
   - thermal velocity
   - particle flux
   - kinetic-energy flux

3. **Electromagnetic environment**
   - prescribed magnetic field
   - ideal-MHD motional electric field
     `E = -v_sw x B`

4. **Particle dynamics**
   - momentum-based relativistic update
   - relativistic Boris-style pusher
   - trajectory and energy diagnostics

5. **Harvester accounting**
   - collector area
   - collection efficiency
   - conversion efficiency
   - estimated usable power

## Important physical limitation

This is **not** a full heliospheric plasma/MHD simulation and it does not
claim that the selected engineering efficiencies are experimentally achieved.
They are explicit model parameters for sensitivity analysis.

The key quantity for an energy-harvesting concept is the available energy
flux. A passive collector cannot continuously extract more energy than is
available in the intercepted particle/field energy budget.

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

## Suggested research extensions

- parameter sweeps over solar-wind density and speed;
- Monte-Carlo proton populations;
- timestep-convergence analysis;
- field-strength sensitivity;
- energy conservation diagnostics;
- comparison with electromagnetic solar irradiance;
- uncertainty propagation;
- dimensionless scaling;
- reproducible CSV output;
- comparison of numerical integrators.

## Reproducibility

All default numerical parameters are explicitly defined in the source files.
The Monte-Carlo generator uses a fixed seed when sampling is requested.
