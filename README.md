# CP Violation Analysis in $\eta' \to \eta \pi \pi$ Decays

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research-orange)

## 📌 Project Overview
This repository implements a **statistical modeling framework** for analyzing Chiral Perturbation Theory ($\chi$PT) constraints in $\eta' \to \eta \pi \pi$ decays. The project focuses on reconstructing the Dalitz plot density using dispersive methods to probe potential **CP-violation** signatures.

The core analysis engine performs a **Maximum Likelihood Estimation (MLE)** to fit theoretical amplitude parameters to experimental data (based on BESIII 2018 datasets), incorporating numerical integration of Omnès functions and rigorous covariance estimation.

## 🚀 Key Features
* **Dispersive $\chi$PT Implementation:** Rigorous modeling of partial wave amplitudes using unitarized Omnès functions.
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
