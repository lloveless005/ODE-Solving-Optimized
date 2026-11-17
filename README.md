The purpose of this project is to solve many systems of ODEs in parallel and compare the performance of the simulation on CPU and on GPU. The ODE system of choice is the Lotka-Volterra model, a simple, non-stiff predator-prey model. Many random parameter sets are generated in order to simulate the process of sensitivity analysis via Saltelli sampling and Sobol analysis. 

# My Hardware

| Part | Model|
| ---- | ---- |
| CPU | AMD Ryzen 7 5800X3D (8c/16t) |
| GPU | Nvidia RTX 5070 |

# Usage

 1. First, set the sample size in ```inc/globals.h```. In my case, my GPU has 12 GB VRAM, so $2^{27}$ samples is the maximum that can fit on the card, but I opt for $2^{26}$.

 2. Create the input and output folders: ```mkdir -p input output```

 3. Build the code with ```make```, and run ```build/std-params``` to generate the parameters for the tests.

 4. Run ```build/lv``` (CPU solver) and ```build/lv-cu``` (GPU solver). Optionally, you can run ```build/lv-nf```, which has no compiler optimizations, but it will take a significant amount of time. The most recently run CPU solver will be used in the comparison script.

 5. Run ```build/comparison``` to see the difference statistics in solver results from CPU to GPU, and GPU FP64 to GPU FP32.

 # Results

 ### Simulation time for $2^{26}$ samples

 Format: avg/min/max/stdev ms (across 10 trials). Two simulations were run initially for warm-up. Does not include file I/O or PCIe transfers. Utilizes all 16 CPU threads.

 | CPU FP64 no-flags (1 trial) | CPU FP64 | GPU FP64 | GPU FP32 |
 | ---- | ---- | ---- | ---- |
 | 97821 ms | 8906/8696/9131/136.9 | 1274/1255/1311/19.49 | 21.43/21.03/22.88/0.5338 |

 - GPU FP64 speedup from CPU FP64: 7x
 - GPU FP32 speedup from CPU FP64: 416x

 ### Numerical differences

 || CPU FP64 vs. GPU FP64 | GPU FP64 vs. GPU FP32 |
 |----|----|----|
 | Max absolute difference | 0 | $7.51 \times 10^{-5}$ |
 | Max relative difference | 0 | $3.08 \times 10^{-6}$ |
 | Average relative difference | 0 | $4.17 \times 10^{-7}$ |

Note that CPU and GPU FP64 solvers produce identical results, but omitting optimizations will produce different results, likely due to the lack of some instructions like fused multiply-add. The CPU solver is compiled with ```-O3 -march=native```, so different CPU architectures may also lead to different results. We see that the average FP32 solve is 59.4x faster than an FP64 solve on the RTX 5070, so in applications where a small error is acceptable, solving with FP32 is significantly faster. (According to Nvidia, on the RTX 5070, FP32 solving should be up to 64x faster than FP64 solving).

# Future work

- optimizing file I/O
- profiling GPU kernel