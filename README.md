The goal for this project is to start with a simple implementation of parallel ODE solving and optimize it as much as possible. I have a simple non-stiff ODE system, the Lotka-Volterra Model, and I am solving this system for many different parameter values. This sort of emulates Saltelli sampling and Sobol analysis, but I don't really want to implement those in C++ just yet.

# Benchmarks for $2^{20}$ simulations

| Optimization | Simulation Time | Total Time |
| ---- | ---- | ---- |
| Start | 13863 ms | 14083 ms |
| Add -O3 compilation flag | 2080 ms | 2115 ms |
| Add -march=native flag | 1910 ms | 1943 ms |
| Add parallelism via OpenMP | 185 ms | 212 ms |
| Condense random parameter generation | 185 ms | 208 ms |
| Add -ffast-math compilation flag | 159 ms | 180 ms |