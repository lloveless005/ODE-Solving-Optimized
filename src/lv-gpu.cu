#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <iostream>
#include <cuda/std/array>
#include <string>
#include <vector>

using namespace std;

using State = cuda::std::array<double, 2>;
using Param = cuda::std::array<double, 4>;

#define NUM_PARAMS 4
#define SAMPLES (1 << 20)
#define DEFAULT_PARAMS cuda::std::array<double, 4>{1.0, 0.1, 0.075, 1.5}

#define THREADS 256

#define cudaCheckError(call) { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
}

#define curandCheckError(call) { \
    curandStatus_t e = call; \
    if (e != CURAND_STATUS_SUCCESS) { \
        fprintf(stderr, "CURAND failure %d at %s:%d\n", e, __FILE__, __LINE__); \
    } \
}




__global__ void SolveGPU(cuda::std::array<double, 2 * SAMPLES>* results, 
                        const double t0, 
                        const double t_end, 
                        const double h, 
                        const cuda::std::array<double, SAMPLES * NUM_PARAMS> param_set);

__global__ void GenerateParameters(double* param_set, const int size);

__device__ State ode_val(const Param &params, const State &inState);

__device__ void solve_ode(double t0, double t_end, State &inState, double h, const Param &inParams);


int main() {
    int errorStatus = 0;

    constexpr double t0 = 0.0;
    constexpr double t_end = 10.0;
    constexpr double h = 0.1;

    // Allocate arrays on GPU
    cuda::std::array<double, SAMPLES * NUM_PARAMS>* param_set = nullptr;        cudaCheckError(cudaMalloc(&param_set, SAMPLES * sizeof(Param)));

    cuda::std::array<double, 2 * SAMPLES>* results = nullptr;
    cudaCheckError(cudaMalloc(&results, SAMPLES * sizeof(State)));

    // Fill param_set array with random uniform doubles

    curandGenerator_t gen;
    curandCheckError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
    curandCheckError(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

    curandCheckError(curandGenerateUniformDouble(gen, param_set->data(), SAMPLES * NUM_PARAMS));

    // Use random uniform values to make varied parameter sets

    int blocksGen = (SAMPLES * NUM_PARAMS + THREADS - 1) / THREADS;
    GenerateParameters<<<blocksGen, THREADS>>>(param_set->data(), SAMPLES * NUM_PARAMS);

    // todo: solve ODEs for random parameter sets
    // todo: copy results array to host

    cudaCheckError(cudaFree(param_set));
    cudaCheckError(cudaFree(results));

    return errorStatus;
}


__global__ void SolveGPU(double* results,  
                        const double t0, 
                        const double t_end, 
                        const double h, 
                        const double* param_set) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SAMPLES) {
        State temp_state = {10.0, 5.0};
        Param temp_params;
        for (int idx = 0; idx < NUM_PARAMS; idx++) {
            temp_params[idx] = param_set[i * NUM_PARAMS + idx];
        }

        solve_ode(t0, t_end, temp_state, h, temp_params);

        results[i * 2] = temp_state[0];
        results[i * 2 + 1] = temp_state[1];
    }
}

__global__ void GenerateParameters(double* param_set, const int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SAMPLES * NUM_PARAMS) {
        param_set[i] = (0.85 + 0.3 * param_set[i]) * DEFAULT_PARAMS[i % 4];
    }
}


__device__ State ode_val(const Param &params, const State &inState) {
    return {
        params[0] * inState[0] - params[1] * inState[0] * inState[1],
        params[2] * inState[0] * inState[1] - params[3] * inState[1]
    };
}

__device__ void solve_ode(double t0, double t_end, State &inState, double h, const Param &inParams) {
    double t = t0;

    while (t < t_end) {
        if (t + h > t_end) {
            h = t_end - t;
        }

        // Runge-Kutta 4th order solver
        State k1 = ode_val(inParams, inState);

        State temp_state = {inState[0] + (h * k1[0]) / 2.0, inState[1] + (h * k1[1]) / 2.0};
        State k2 = ode_val(inParams, temp_state);

        temp_state = {inState[0] + (h * k2[0]) / 2.0, inState[1] + (h * k2[1]) / 2.0};
        State k3 = ode_val(inParams, temp_state);

        temp_state = {inState[0] + (h * k3[0]), inState[1] + (h * k3[1])};
        State k4 = ode_val(inParams, temp_state);

        inState[0] += (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * (h / 6.0);
        inState[1] += (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * (h / 6.0);
        t += h;
    }
}