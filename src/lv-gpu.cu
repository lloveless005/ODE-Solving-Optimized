#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <cstdio>
#include <cuda/std/array>
#include <string>
#include <vector>
#include <fstream>
#include <globals.h>
#include <writefile.h>
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

#define INPUT_FILE_FP64 "input/standard-parameters-fp64.bin"
#define INPUT_FILE_FP32 "input/standard-parameters-fp32.bin"
const string OUTPUT_FILE_FP64 = "output/gpu-64-" + to_string(static_cast<int>(log2(SAMPLES))) + ".bin";
const string OUTPUT_FILE_FP32 = "output/gpu-32-" + to_string(static_cast<int>(log2(SAMPLES))) + ".bin";

#define THREADS 256

#define cudaCheckError(call) { \
    cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA failure %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    } \
}

#define cudaCheckKernel() cudaCheckError(cudaGetLastError())


template <typename T>
__device__ State<T> ode_val(const Param<T> &params, const State<T> &inState) {
    return {
        params[0] * inState[0] - params[1] * inState[0] * inState[1],
        params[2] * inState[0] * inState[1] - params[3] * inState[1]
    };
}

template <typename T>
__device__ void solve_ode(int steps, State<T> &inState, T h, const Param<T> &inParams) {
    int current = 0;

    // Constants to avoid template issue

    T two = static_cast<T>(2.0);
    T six = static_cast<T>(6.0);

    while (current < steps) {

        // Runge-Kutta 4th order solver
        
        State<T> k1 = ode_val(inParams, inState);

        State<T> temp_state = {inState[0] + (h * k1[0]) / two, inState[1] + (h * k1[1]) / two};
        State<T> k2 = ode_val(inParams, temp_state);

        temp_state = {inState[0] + (h * k2[0]) / two, inState[1] + (h * k2[1]) / two};
        State<T> k3 = ode_val(inParams, temp_state);

        temp_state = {inState[0] + (h * k3[0]), inState[1] + (h * k3[1])};
        State<T> k4 = ode_val(inParams, temp_state);

        inState[0] += (k1[0] + two * k2[0] + two * k3[0] + k4[0]) * (h / six);
        inState[1] += (k1[1] + two * k2[1] + two * k3[1] + k4[1]) * (h / six);
        current++;
        
    }
}

template <typename T>
__global__ void SolveGPU(T* results, const int steps, const T h, const T* param_set) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < SAMPLES) {
        State<T> temp_state = {10.0, 5.0};
        Param<T> temp_params;
        for (int idx = 0; idx < NUM_PARAMS; idx++) {
            temp_params[idx] = param_set[i * NUM_PARAMS + idx];
        }

        solve_ode(steps, temp_state, h, temp_params);

        results[i * 2] = temp_state[0];
        results[i * 2 + 1] = temp_state[1];
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

    double* device_results_fp64 = nullptr;
    cudaCheckError(cudaMalloc(&device_results_fp64, SAMPLES * 2 * sizeof(double)));

    float* device_results_fp32 = nullptr;
    cudaCheckError(cudaMalloc(&device_results_fp32, SAMPLES * 2 * sizeof(float)));

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
    cout << endl << "Parameter read time: " << param_ms << " ms" << endl;

    if (error_status == 0) {


        //
        // FP64
        //


        // CUDA events for benchmarking ODE solving

        cudaEvent_t solve_fp64_start, solve_fp64_stop;
        cudaCheckError(cudaEventCreate(&solve_fp64_start));
        cudaCheckError(cudaEventCreate(&solve_fp64_stop));

        cudaCheckError(cudaEventRecord(solve_fp64_start));

        // Solving ODEs for each parameter set

        int blocksSolve = (SAMPLES + THREADS - 1) / THREADS;
        SolveGPU<<<blocksSolve, THREADS>>>(device_results_fp64, steps, h, param_set_fp64);
        cudaCheckKernel();

        // Copy results array to host

        vector<double> host_results_fp64(SAMPLES * 2);
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaMemcpy(host_results_fp64.data(), device_results_fp64, SAMPLES * 2  *sizeof(double), cudaMemcpyDeviceToHost));

        // CUDA events for benchmarking ODE solving

        cudaCheckError(cudaEventRecord(solve_fp64_stop));
        cudaCheckError(cudaEventSynchronize(solve_fp64_stop));
        float solve_fp64_ms = 0.0;
        cudaCheckError(cudaEventElapsedTime(&solve_fp64_ms, solve_fp64_start, solve_fp64_stop));
        cout << "FP64 simulation time: " << solve_fp64_ms << " ms" << endl;

        //
        // FP32
        //


        // CUDA events for benchmarking ODE solving

        cudaEvent_t solve_fp32_start, solve_fp32_stop;
        cudaCheckError(cudaEventCreate(&solve_fp32_start));
        cudaCheckError(cudaEventCreate(&solve_fp32_stop));

        cudaCheckError(cudaEventRecord(solve_fp32_start));

        // Solving ODEs for each parameter set

        SolveGPU<<<blocksSolve, THREADS>>>(device_results_fp32, steps, static_cast<float>(h), param_set_fp32);
        cudaCheckKernel();

        // Copy results array to host

        vector<float> host_results_fp32(SAMPLES * 2);
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaMemcpy(host_results_fp32.data(), device_results_fp32, SAMPLES * 2  *sizeof(float), cudaMemcpyDeviceToHost));

        // CUDA events for benchmarking ODE solving

        cudaCheckError(cudaEventRecord(solve_fp32_stop));
        cudaCheckError(cudaEventSynchronize(solve_fp32_stop));
        float solve_fp32_ms = 0.0;
        cudaCheckError(cudaEventElapsedTime(&solve_fp32_ms, solve_fp32_start, solve_fp32_stop));
        cout << "FP32 simulation time: " << solve_fp32_ms << " ms" << endl;


        // Output results to files

        cudaEvent_t write_start, write_stop;
        cudaCheckError(cudaEventCreate(&write_start));
        cudaCheckError(cudaEventCreate(&write_stop));

        cudaCheckError(cudaEventRecord(write_start));

        ofstream outputFile64;
        outputFile64.open(OUTPUT_FILE_FP64, ios::binary);

        ofstream outputFile32;
        outputFile32.open(OUTPUT_FILE_FP32, ios::binary);

        if (!outputFile64.is_open()) {
            cout << "Failed to open " << OUTPUT_FILE_FP64 << endl;
            error_status = 3; 
        } else if (!outputFile32.is_open()) {
            cout << "Failed to open " << OUTPUT_FILE_FP32 << endl;
            error_status = 4;
        } else {

            WriteFile(outputFile64, host_results_fp64);
            WriteFile(outputFile32, host_results_fp32);

        }

        outputFile64.close();
        outputFile32.close();

        // CUDA events for file writing

        cudaCheckError(cudaEventRecord(write_stop));
        cudaCheckError(cudaEventSynchronize(write_stop));
        float write_ms = 0.0;
        cudaCheckError(cudaEventElapsedTime(&write_ms, solve_fp32_start, write_stop));
        cout << "Write time: " << write_ms << " ms" << endl;

    }

    // CUDA events for total program benchmark 

    cudaCheckError(cudaEventRecord(total_stop));
    cudaCheckError(cudaEventSynchronize(total_stop));
    float total_ms = 0.0;
    cudaCheckError(cudaEventElapsedTime(&total_ms, total_start, total_stop));
    cout << endl << "Total time: " << total_ms << " ms" << endl;

    // Deallocate GPU arrays

    cudaCheckError(cudaFree(param_set_fp64));
    cudaCheckError(cudaFree(param_set_fp32));
    cudaCheckError(cudaFree(device_results_fp64));
    cudaCheckError(cudaFree(device_results_fp32));

    return error_status;
}