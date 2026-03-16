# My attempt at Spiking Neural Network (SNN) Engine

**Author:** Rudraneel Shee | B.Tech Engineering Physics
**Focus:** Trying to learn about Neuromorphic Computing

## Project Overview
**ONGOING PROJECT**
This repository contains currently contains my files to build a Spiking Neural Network (SNN) from scratch in C++. Rather than utilizing high-level machine learning frameworks, this project approaches neuromorphic computing from a fundamental physics and dynamical systems perspective. 

## Development Architecture
I am documenting the project in the same structure I am building step by step, to ensure mathematical accuracy and computational efficiency before scaling:

* **`01_Single_LIF_Neuron/`** - Core physics engine modeling continuous RC circuit dynamics and discontinuous threshold resets.
* **`02_Vectorized_Layer/`** - *(In Progress)* High-performance network scaling using the C++ Eigen template library for matrix operations.
* **`03_STDP_Learning/`** - *(Planned)* Implementation of localized Spike-Timing-Dependent Plasticity for unsupervised temporal learning.

## Part 1: The Single LIF Neuron
The foundational unit of this network is the Leaky Integrate-and-Fire (LIF) neuron. It is modeled as a parallel RC circuit where the cell membrane acts as a capacitor and ion channels act as resistors.

### The Physics & Mathematics
The continuous subthreshold membrane potential `V(t)` is governed by the first-order differential equation:
`τ_m * (dV/dt) = -(V(t) - V_rest) + R * I_in(t)`

Because the simulation operates in discrete computational steps (`dt`), the continuous derivative is approximated using the **Forward Euler method**:
`V(t+dt) = V(t) + (dt / τ_m) * [ -(V(t) - V_rest) + R * I_in(t) ]`

Once the voltage crosses a defined threshold (`V_th`), the continuous integration is interrupted. A discrete spike event is recorded, and the voltage is instantaneously forced back to the reset potential (`V_reset`), creating the characteristic sawtooth wave.

### Visualization
![LIF Dynamics](01_Single_LIF_Neuron/lif_graph.png)

### Build & Run Instructions (Linux/Ubuntu)
To compile the single neuron physics engine,
```bash
cd 01_Single_LIF_Neuron
g++ single_lif.cpp -o single_lif
./single_lif
```

Enter required input

For visualization, run
```bash
python3 plot_lif.py
```
