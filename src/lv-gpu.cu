#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <iostream>
#include <cstdio>
#include <cuda/std/array>
#include <string>
#include <vector>
#include <fstream>
#include <globals.h>
#include <cmath>

/*TODO
- switch file IO to binary for fast reads
- make a program that rigorously compares the deviation from the cpu script
- investigate fp32 vs fp64
- profile the program to find more optimizations
*/

using namespace std;

template <typename T>
using State = cuda::std::array<T, 2>;

template <typename T>
using Param = cuda::std::array<T, 4>;

#define INPUT_FILE_FP64 "input/standard-parameters-fp64.txt"
#define INPUT_FILE_FP32 "input/standard-parameters-fp32.txt"

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

#define cudaCheckKernel() cudaCheckError(cudaGetLastError())



template <typename T>
__global__ void SolveGPU(T* results, const int steps, const T h, const T* param_set) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SAMPLES) {
        State<T> temp_state = {10.0, 5.0};
        Param<T> temp_params;
        for (int idx = 0; idx < NUM_PARAMS; idx++) {
            temp_params[idx] = param_set[i * NUM_PARAMS + idx];
        }

        solve_64_ode(steps, temp_state, h, temp_params);

        results[i * 2] = temp_state[0];
        results[i * 2 + 1] = temp_state[1];
    }
}

template <typename T>
__device__ State<T> ode_val(const Param<T> &params, const State<T> &inState) {
    return {
        params[0] * inState[0] - params[1] * inState[0] * inState[1],
        params[2] * inState[0] * inState[1] - params[3] * inState[1]
    };
}

template <typename T>
__device__ void solve_64_ode(int steps, State<T> &inState, T h, const Param<T> &inParams) {
    int current = 0;

    while (current < steps) {

        // Runge-Kutta 4th order solver
        
        State<T> k1 = ode_val(inParams, inState);

        State<T> temp_state = {inState[0] + (h * k1[0]) / 2.0, inState[1] + (h * k1[1]) / 2.0};
        State<T> k2 = ode_val(inParams, temp_state);

        temp_state = {inState[0] + (h * k2[0]) / 2.0, inState[1] + (h * k2[1]) / 2.0};
        State<T> k3 = ode_val(inParams, temp_state);

        temp_state = {inState[0] + (h * k3[0]), inState[1] + (h * k3[1])};
        State<T> k4 = ode_val(inParams, temp_state);

        inState[0] += (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * (h / 6.0);
        inState[1] += (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * (h / 6.0);
        current++;
        
    }
}


int main() {
    int error_status = 0;

    constexpr double t0 = 0.0;
    constexpr double t_end = 10.0;
    constexpr double h = 0.1;
    constexpr int steps = ceil((t_end - t0) / h);

    // CUDA events for total program benchmark 

    cudaEvent_t total_start, total_stop;
    cudaCheckError(cudaEventCreate(&total_start));
    cudaCheckError(cudaEventCreate(&total_stop));

    cudaCheckError(cudaEventRecord(total_start));

    // CUDA events for benchmarking parameter generation

    cudaEvent_t param_start, param_stop;
    cudaCheckError(cudaEventCreate(&param_start));
    cudaCheckError(cudaEventCreate(&param_stop));

    cudaCheckError(cudaEventRecord(param_start));

    // Allocate arrays on GPU

    double* param_set_fp64 = nullptr;        
    cudaCheckError(cudaMalloc(&param_set_fp64, SAMPLES * NUM_PARAMS * sizeof(double)));

    float* param_set_fp32 = nullptr;
    cudaCheckError(cudaMalloc(&param_set_fp32, SAMPLES * NUM_PARAMS * sizeof(float)));

    double* results = nullptr;
    cudaCheckError(cudaMalloc(&results, SAMPLES * 2 * sizeof(double)));

    // Fill param_set from file

    ifstream inputFile64;
    inputFile64.open(INPUT_FILE_FP64, ios::binary);

    ifstream inputFile32;
    inputFile32.open(INPUT_FILE_FP32, ios::binary);

    if (!inputFile64.is_open()) {
        cout << "Failed to open " << INPUT_FILE_FP64 << endl;
        error_status = 1;
    } else if (!inputFile32.is_open()) {
        cout << "Failed to open " << INPUT_FILE_FP32 << endl;
        error_status = 2;
    } else {
        vector<double> host_params_fp64(SAMPLES * NUM_PARAMS);
        vector<float> host_params_fp32(SAMPLES * NUM_PARAMS);
        
        inputFile64.read(reinterpret_cast<char*>(host_params_fp64.data()), SAMPLES * NUM_PARAMS * sizeof(double));
        inputFile32.read(reinterpret_cast<char*>(host_params_fp32.data()), SAMPLES * NUM_PARAMS * sizeof(float));

        cudaCheckError(cudaMemcpy(param_set_fp64, host_params_fp64.data(), SAMPLES * NUM_PARAMS * sizeof(double), cudaMemcpyHostToDevice));
        cudaCheckError(cudaMemcpy(param_set_fp32, host_params_fp32.data(), SAMPLES * NUM_PARAMS * sizeof(float), cudaMemcpyHostToDevice));

        inputFile64.close();
        inputFile32.close();
    }

    // CUDA events for benchmarking parameter generation

    cudaCheckError(cudaEventRecord(param_stop));
    cudaCheckError(cudaEventSynchronize(param_stop));
    float param_ms = 0.0;
    cudaCheckError(cudaEventElapsedTime(&param_ms, param_start, param_stop));
    cout << "Parameter generation time: " << param_ms << " ms" << endl;

    if (error_status == 0) {
        // CUDA events for benchmarking ODE solving

        cudaEvent_t solve_64_start, solve_64_stop;
        cudaCheckError(cudaEventCreate(&solve_64_start));
        cudaCheckError(cudaEventCreate(&solve_64_stop));

        cudaCheckError(cudaEventRecord(solve_64_start));

        // Solving ODEs for each parameter set

        int blocksSolve = (SAMPLES + THREADS - 1) / THREADS;
        SolveGPU<<<blocksSolve, THREADS>>>(results, t0, t_end, h, param_set_fp64);
        cudaCheckKernel();

        // Copy results array to host

        vector<double> host_results(SAMPLES * 2);
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaMemcpy(host_results.data(), results, SAMPLES * 2  *sizeof(double), cudaMemcpyDeviceToHost));

        // CUDA events for benchmarking ODE solving

        cudaCheckError(cudaEventRecord(solve_64_stop));
        cudaCheckError(cudaEventSynchronize(solve_64_stop));
        float solve_64_ms = 0.0;
        cudaCheckError(cudaEventElapsedTime(&solve_64_ms, solve_64_start, solve_64_stop));
        cout << "Simulation time: " << solve_64_ms << " ms" << endl;

        // Output the head of the states set

        for (int idx = 0; idx < 5; idx++) {
            cout << "State " << idx << ": " << host_results[idx * 2] << " " << host_results[idx * 2 + 1] << endl;
        }
    }

    // CUDA events for total program benchmark 

    cudaCheckError(cudaEventRecord(total_stop));
    cudaCheckError(cudaEventSynchronize(total_stop));
    float total_ms = 0.0;
    cudaCheckError(cudaEventElapsedTime(&total_ms, total_start, total_stop));
    cout << "Total time: " << total_ms << " ms" << endl;

    // Deallocate GPU arrays

    cudaCheckError(cudaFree(param_set_fp64));
    cudaCheckError(cudaFree(param_set_fp32));
    cudaCheckError(cudaFree(results));

    return error_status;
}