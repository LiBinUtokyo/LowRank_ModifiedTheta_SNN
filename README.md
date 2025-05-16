

# Low-Rank SNNs with Gamma Oscillations

This repository contains the simulation and analysis code for the paper:

**"Neural oscillation in low-rank SNNs: bridging network dynamics and cognitive function"**  
by Bin Li, et al.  
Accepted in *Frontiers in Computational Neuroscience*, 2025.

---

## 🔍 Overview

This project investigates how the phase of gamma oscillations in low-rank spiking neural networks (SNNs) modulates cognitive task performance.  
We combine biophysically grounded modeling and dynamical systems analysis to bridge network structure and function.

**Key components:**
- Biophysically realistic SNNs based on the voltage-dependent theta neuron model
- Macroscopic model bifurcation analysis
- Go-Nogo task simulations under oscillatory and stationary states
- Phase-dependent performance evaluation

---

## 🧠 Main Features

- Predictable network-level dynamics via bifurcation analysis
- Phase-specific modulation of task output
- Low-rank structure in E→E connections
- ING-type gamma oscillation reproduction
- Strict implementation of Dale’s principle

---

## 📁 Folder Structure
- `bifurcation/voltage_dependent_theta.ode`: ode file for bifurcation analysis (Software: XPPaut)
- `low-rank SNN`: code for low-rank spiking neural networks
    - 'configures/': configuration files
    - 'functions.py': functions for SNN
    - 'lowranksnn.py': code of module for low-rank SNN
    - 'main_pytorch.ipynb': main code for low-rank SNN (using PyTorch)
    - 'phase_sensitivity.ipynb': code for phase sensitivity analysis

---
## 📄 License

This project is licensed under the MIT License.

---
## 📬 Contact

For questions, please contact:

📧 li [at] neuron.t.u-tokyo.ac.jp

🔗 Academic Homepage: https://libinutokyo.github.io/



