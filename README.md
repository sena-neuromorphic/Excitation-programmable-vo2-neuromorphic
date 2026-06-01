# Excitation-programmable-vo2-neuromorphic

This repository contains the code for the paper:

*Excitation programmable electrothermal hysteretic operating regimes in VO₂ devices *  


## Abstract

This repository contains the Python implementation of an excitation-programmable electrothermal VO₂ model based on the Limiting Loop Proximity (LLP) hysteresis framework. The model separates intrinsic thermal hysteresis from excitation-dependent electrical dynamics, enabling the simulation of voltage-driven and current-driven operating regimes within the same material platform. It reproduces hysteretic switching, negative differential resistance (NDR) and electrothermal feedback effects relevant to memristive, neuristive, and neuromorphic applications.

## Notes

The code was originally developed and tested in a Google Colab environment. Running it in Colab is recommended for faster and smoother execution.


## Main Scripts

The following scripts reproduce the results presented in the paper:

- `Figure4_5_6__main_article.py`  
  Reproduces Figures 4, 5, and 6 from the main article. The script also generates additional figures currently under evaluation for inclusion either in the main manuscript, the Supplementary Information, or future versions of the work.
  
  All provided scripts can be used to model VO$_2$ films using the explicit electrothermal LLP formulation. For questions, please contact the authors. If you use this code, please cite the associated publication.
  If you use the LLP model to study other metal–insulator transition materials, feel free to share your results with us — we would be happy to hear about your work.

MIT License
Copyright (c) 2026 Bruno A. S. F. Sena and Luiz A. L. de Almeida

  ## Requirements

The code was developed and tested in a Google Colab environment.

All required dependencies are listed in the `requirements.txt` file. These packages are available by default in Google Colab, enabling straightforward and reproducible execution.

## Overview

The purpose of this code is to **disentangle intrinsic thermal hysteresis from excitation-dependent electrical switching behavior** in VO₂ devices. The framework explicitly separates:

- Intrinsic thermal hysteresis associated with the metal–insulator transition
- Electrical manifestations shaped by external excitation and circuit constraints

By preserving identical material parameters, the simulations isolate the role of the **excitation mode** in shaping the observed current–voltage hysteresis.

---

## Implemented excitation modes

Two canonical electrical driving schemes are implemented:

- **Force–V (Thevenin drive)**  
  Voltage source with series resistance, emphasizing load-line effects and threshold-like switching.

- **Force–I (Norton drive)**  
  Current-driven excitation that constrains the electrical trajectory and exposes the intrinsic electro–thermal response.

These modes correspond directly to the configurations analyzed and compared in the article.

---

## Model features

- Coupled electrical and thermal state equations
- Limiting Loop Proximity (LLP) hysteresis operator
- Event-driven hysteresis updates triggered exclusively by thermal reversals
- Solver-agnostic formulation compatible with time-domain ODE integration
- Reproducible separation between material properties and circuit-induced effects

The hysteresis state is held fixed between reversal events and only evaluated during integration, ensuring causal and solver-independent behavior.

---

## How to run

The simulations are implemented in **Python** and designed to run in a standard Jupyter environment.

Typical workflow:
1. Select the excitation mode (Force–V or Force–I)
2. Define material, thermal, and circuit parameters
3. Run the time-domain simulation
4. Generate time-domain traces, phase portraits, and I–V hysteresis loops

All parameters used in the figures of the article are explicitly reported to ensure reproducibility.

---

## Reproducibility and scope

While demonstrated here in a Python-based environment, the formulation is independent of the numerical solver and can be translated to other simulation platforms, including circuit-oriented frameworks.

---

## License

This project is released under the **MIT License**.  
See the `LICENSE` file for details.

---

## Citation

If you use this code in academic work, please cite the accompanying article.
