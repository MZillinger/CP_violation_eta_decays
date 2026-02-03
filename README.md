# Phenomenology of $\eta' \to \eta \pi \pi$ Decays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

## 📌 Project Overview
This repository implements a **statistical modeling framework** for analyzing an improved version of the large Nc Chiral Perturbation Theory framework for the decay $\eta' \to \eta \pi \pi$. This analysis focuses on reconstructing the Dalitz plot density as well as the branching fraction from the PDG. Moreover, this analysis is related to the $CP$-violating [project](https://arxiv.org/abs/2210.14925) of $\eta^{(\prime)}$ decays.

The core analysis engine performs a **Maximum Likelihood Estimation (MLE)** to fit theoretical amplitude parameters to experimental data (based on [BESIII 2018](https://arxiv.org/abs/1709.04627) 2018 datasets), incorporating numerical integration of Omnès functions and rigorous covariance estimation.

## 🚀 Key Features
* **Implementation of unitarized amplitude:** Rigorous modeling of partial wave amplitudes with unitarization methods.
* **Numerical Integration Pipeline:** Automated pre-computation of phase-space integrals using `scipy.integrate.dblquad` for computational efficiency.
* **Statistical Fitting Engine:**
    * Simultaneous $\chi^2$ minimization of Dalitz plot bins and decay rate constraints.
    * Weighted event handling for experimental efficiency corrections.
* **Error Analysis:**
    * Hessian matrix computation for parameter uncertainty estimation.
    * Correlation matrix visualization for detecting parameter degeneracies.

## 📂 Repository Structure
```text
.
├── data/                       # Experimental data and theoretical inputs
│   ├── Omnesfunctions-d00...   # Dispersive phase shift inputs
│   └── BESIII-2018...          # Dalitz plot event data
├── etap_eta_pipi_model.py      # Core Model Class (Physics Logic)
├── fitting_etap_eta_pipi.ipynb # Jupyter Notebook for Visualization & Plots
└── README.md                   # Project Documentation

## 🏗️ Code Architecture

This project separates the physical theory from the statistical analysis to ensure modularity:

* **`etaprime_model.py` (The Physics Engine)**
    * **Role:** Acts as the backend library.
    * **Responsibilities:** Implements the core mathematical framework, including the interpolation of Omnès functions, numerical integration of dispersion relations, and construction of partial wave amplitudes.
    * *Usage:* Can be imported as a module or run as a script for quick CLI validation.

* **`analysis.ipynb` (The Analysis Dashboard)**
    * **Role:** Acts as the interactive frontend.
    * **Responsibilities:** The statistical fitting procedure with manuell minimization of $\chi^{2}$ as well as the usage of the package [iminuit](https://pypi.org/project/iminuit/), calculation of correlation matrices, and generation of 2D residuals in a Dalitz plot format.
