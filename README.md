1D Arterial Hemodynamics Solver

This repository contains a high-performance 1D solver for simulating blood flow and pressure dynamics across the human arterial tree.

The solver uses a reduced-order model based on the 1D Navier-Stokes equations (conservation of mass and momentum) to capture pulse wave propagation, reflections at bifurcations, and fluid-structure interaction in elastic vessels. It is designed to be computationally efficient, allowing for the rapid generation of large cohorts of "virtual patients."
Key Features

  Physiological Modeling: Simulates wave propagation in compliant arteries.

  Network Capable: Handles complex arterial trees with multiple bifurcations.

  Efficient: Optimized for fast execution, suitable for generating synthetic datasets (e.g., for Machine Learning training).

  Configurable: Allows easy modification of boundary conditions (inlet flow, outlet resistance) and vessel properties (stiffness, diameter).
