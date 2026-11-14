The goal for this project is to start with a simple implementation of parallel ODE solving and optimize it as much as possible. I have a simple non-stiff ODE system, the Lotka-Volterra Model, and I am solving this system for many different parameter values. This sort of emulates Saltelli sampling and Sobol analysis, but I don't really want to implement those in C++ just yet. I am also adding a GPU accelerated version of the code and comparing the two methods.

# Hardware

| Part | Model|
| ---- | ---- |
| CPU | AMD Ryzen 7 5800X3D |
| GPU | Nvidia RTX 5070 |
