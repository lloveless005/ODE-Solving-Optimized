#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cuda/std/array>
#include <string>

using namespace std;

using State = cuda::std::array<double, 2>;
using Param = cuda::std::array<double, 4>;

#define num_params 4


__global__ void SolveGPU(State results[], const int samples, 
                        const double t0, const double t_end, const double h, const Param param_set[]);

__device__ State ode_val(const Param &params, const State &nState);

__device__ void solve_ode(double t0, double t_end, State &inState, double h, const Param &inParams);


int main(int argc, char *argv[]) {
    int errorStatus = 0;
    if (argc < 2) errorStatus = 1;
    else if (stoi(argv[1]) > 24) errorStatus = 2;
    else {

    }

    return errorStatus;
}


__global__ void SolveGPU(State results[], const int samples, 
                        const double t0, const double t_end, const double h, const Param param_set[]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < samples) {
        State temp_state = {10.0, 5.0};
        Param temp_params = param_set[i];

        solve_ode(t0, t_end, temp_state, h, temp_params);
        results[i] = temp_state;
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