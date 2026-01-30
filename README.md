# Bachelorprojekt
This is for product/scientific development for my bachelorprojekt.

## Optimizing nHesel Plasma Simulations using Scientific Machine Learning

Bachelor project at the Technical University of Denmark (DTU)

**Student:**        Andreas Bjarnastein Antoft
**Supervisor\*:**   Jesper Løve Hinrich         https://orbit.dtu.dk/en/persons/jesper-løve-hinrich/
**Supervisor:**     Morten Mørup                https://orbit.dtu.dk/en/persons/morten-mørup/
**Co-supervisor:**  Alexander Simon Thrysøe     https://orbit.dtu.dk/en/persons/alexander-simon-thrysøe/
**Period:**         Spring 2026

\* Jesper is not certified to be supervisor thus we have Morten. Jesper will mainly guide me and Morten will make sure everythin is in order.
---

## 📌 Project Overview

Numerical simulations of plasma edge dynamics are a cornerstone of fusion research, but solving the underlying nonlinear partial differential equations (PDEs) at high spatial and temporal resolution is computationally expensive.

This bachelor project investigates whether **Scientific Machine Learning (SciML)** methods can be used to accelerate or approximate **2D nHesel plasma simulations** without significant loss of physical accuracy. In particular, the project focuses on **Physics-Informed Neural Networks (PINNs)**, where the governing PDEs are explicitly embedded into the learning process.

The goal is to compare classical numerical simulation methods with ML-based surrogate or acceleration models in terms of:
- Accuracy  
- Stability  
- Computational cost  

---

## 🧠 Scientific Approach

The physical system is described by a set of coupled PDEs:

\[
u_i(x, t), \quad i = 1, \dots, n
\]

These solutions are approximated using a parameterized neural network:

\[
u_i(x, t) \approx f_\theta(x, t)
\]

where \( f_\theta \) is trained using physics-informed loss functions incorporating:
- Governing PDEs  
- Initial conditions  
- Boundary conditions  

---

## 📂 Repository Structure

```text
.
├── Tekst_filer                 # Gathering all text files
    ├── For_arbejde.txt         # Diary to keep trak of everythin
    ├── indberette_projekt.txt  # Pre report of the project
├── LICENCE     #Standard MIT licence
└── README.md   #This beauty of a file :)
