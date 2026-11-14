#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand.h>
#include <iostream>
#include <cstdio>
#include <cuda/std/array>
#include <string>
#include <vector>
#include <fstream>

using namespace std;

using State = cuda::std::array<double, 2>;
using Param = cuda::std::array<double, 4>;

#define NUM_PARAMS 4
#define SAMPLES (1 << 20)

#define INPUT_FILE "input/standard-parameters.txt"
constexpr bool use_input_file = false;

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




__global__ void SolveGPU(double* results,  
                        const double t0, 
                        const double t_end, 
                        const double h, 
                        const double* param_set);

__global__ void GenerateParameters(double* param_set, const int size);

__device__ State ode_val(const Param &params, const State &inState);

__device__ void solve_ode(double t0, double t_end, State &inState, double h, const Param &inParams);


int main() {
    int error_status = 0;

    constexpr double t0 = 0.0;
    constexpr double t_end = 10.0;
    constexpr double h = 0.1;

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

    double* param_set = nullptr;        
    cudaCheckError(cudaMalloc(&param_set, SAMPLES * NUM_PARAMS * sizeof(double)));

    double* results = nullptr;
    cudaCheckError(cudaMalloc(&results, SAMPLES * 2 * sizeof(double)));

    if (use_input_file) {

        // Fill param_set from file

        ifstream inputFile;
        inputFile.open(INPUT_FILE);

        if (!inputFile.is_open()) {
            cout << "Failed to open " << INPUT_FILE << endl;
            error_status = 1;
        } else {
            double* host_params = new double[SAMPLES * NUM_PARAMS];
            double temp;
            int position = 0;

            while (inputFile >> temp) {
                host_params[position] = temp;
                position++;
            }

            cudaCheckError(cudaMemcpy(param_set, host_params, SAMPLES * NUM_PARAMS * sizeof(double), cudaMemcpyHostToDevice));
        }

    } else {

        // Fill param_set array with random uniform doubles
     
        curandGenerator_t gen;
        curandCheckError(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10));
        curandCheckError(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

        curandCheckError(curandGenerateUniformDouble(gen, param_set, SAMPLES * NUM_PARAMS));

        // Use random uniform values to make varied parameter sets

        int blocksGen = (SAMPLES * NUM_PARAMS + THREADS - 1) / THREADS;
        GenerateParameters<<<blocksGen, THREADS>>>(param_set, SAMPLES * NUM_PARAMS);
        cudaCheckKernel();

        curandCheckError(curandDestroyGenerator(gen));

    }

    cudaCheckError(cudaEventRecord(param_stop));
    cudaCheckError(cudaEventSynchronize(param_stop));
    float param_ms = 0.0;
    cudaCheckError(cudaEventElapsedTime(&param_ms, param_start, param_stop));
    cout << "Parameter generation time: " << param_ms << " ms" << endl;

    // CUDA events for benchmarking ODE solving
    if (error_status == 0) {
        cudaEvent_t solve_start, solve_stop;
        cudaCheckError(cudaEventCreate(&solve_start));
        cudaCheckError(cudaEventCreate(&solve_stop));

        cudaCheckError(cudaEventRecord(solve_start));

        // Solving ODEs for each parameter set

        int blocksSolve = (SAMPLES + THREADS - 1) / THREADS;
        SolveGPU<<<blocksSolve, THREADS>>>(results, t0, t_end, h, param_set);
        cudaCheckKernel();

        // Copy results array to host

        vector<double> host_results(SAMPLES * 2);
        cudaCheckError(cudaDeviceSynchronize());
        cudaCheckError(cudaMemcpy(host_results.data(), results, SAMPLES * 2  *sizeof(double), cudaMemcpyDeviceToHost));

        cudaCheckError(cudaEventRecord(solve_stop));
        cudaCheckError(cudaEventSynchronize(solve_stop));
        float solve_ms = 0.0;
        cudaCheckError(cudaEventElapsedTime(&solve_ms, solve_start, solve_stop));
        cout << "Simulation time: " << solve_ms << " ms" << endl;

        for (int idx = 0; idx < 5; idx++) {
            cout << "State " << idx << ": " << host_results[idx * 2] << " " << host_results[idx * 2 + 1] << endl;
        }
    }

    cudaCheckError(cudaEventRecord(total_stop));
    cudaCheckError(cudaEventSynchronize(total_stop));
    float total_ms = 0.0;
    cudaCheckError(cudaEventElapsedTime(&total_ms, total_start, total_stop));
    cout << "Total time: " << total_ms << " ms" << endl;

    // Deallocate
    cudaCheckError(cudaFree(param_set));
    cudaCheckError(cudaFree(results));

    return error_status;
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
    constexpr Param defaults{1.0, 0.1, 0.075, 1.5};

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        param_set[i] = (0.85 + 0.3 * param_set[i]) * defaults[i % 4];
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