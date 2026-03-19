# Spiking Neural Network (SNN) Engine attempt

**Author:** Rudraneel Shee | B.Tech Engineering Physics
**Focus:** Trying to learn about Neuromorphic Computing

## Project Overview
**ONGOING PROJECT**
This repository contains currently contains my files to build a Spiking Neural Network (SNN) from scratch in C++. Rather than utilizing high-level machine learning frameworks, this project approaches neuromorphic computing from a fundamental physics and dynamical systems perspective. 

## Development Architecture
I am documenting the project in the same structure I am building step by step, to ensure mathematical accuracy and computational efficiency before scaling:

* **`01_Single_LIF_Neuron/`** - Core physics engine modeling continuous RC circuit dynamics and discontinuous threshold resets.
* **`02_Vectorized_Layer/`** - Implementing multiple neuron network with multiple channels using the C++ Eigen template library for matrix operations.
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
----

## Part 2: The Vectorized SNN Layer

This scales the isolated LIF neuron into a fully connected, parallel processing layer. To handle the increase in computational complexity, the physics engine logic is now vectorized using the C++ **Eigen** library, enabling highly optimized matrix linear algebra.

### The Physics & Mathematics
#### Spatial Integration (The Wiring)

Instead of a single injected current, neurons now receive discrete spikes from multiple input channels. The physical synapses connecting the inputs to the neurons are represented by a weight matrix $\mathbf{W}$. The total incoming current $\mathbf{I}_{in}$ for the entire layer is calculated simultaneously via matrix-vector multiplication:

$$\mathbf{I}_{in} = \mathbf{W} \cdot \mathbf{S}_{in}$$

*(Where $\mathbf{S}_{in}$ is a binary vector representing which sensory inputs fired at the current millisecond).*

#### Temporal Integration & Asynchronous Dynamics

The Forward Euler differential equation is applied across the entire state vector of membrane potentials simultaneously.

To test the layer's biological realism, the simulation drives the network with 5 distinct sensory inputs firing at prime-number frequencies (e.g., **500 Hz, 333 Hz, 142 Hz**). This creates a chaotic, non-repeating interference pattern. Because the initial synaptic weight matrix is randomized, each neuron organically develops a unique "receptive field," demonstrating native **spatiotemporal feature extraction** - different neurons learn to spike only when specific combinations of frequencies overlap.

### Build & Run Instructions (Linux/Ubuntu)

```bash
cd 02_Vectorized_Layer
g++ -I/usr/include/eigen3 snn_layer.cpp -o snn_layer
./snn_layer
```
*(Note: If your Eigen installation is in a different directory, adjust the `-I` flag accordingly).*
