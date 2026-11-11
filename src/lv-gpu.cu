#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

using State = array<double, 2>;
using Param = array<double, 4>;


__global__ void SolveGPU() {
    
}

State ode_val(const Param params, const State states);

void solve_ode(double t0, double t_end, State& states, double h, const Param& params);


int main(int argc, char *arg[]) {
    int errorStatus = 0;
    if (stoi(argv[1] > 24)) errorStatus = 1;
    else {

    }
}


State ode_val(const Param params, const State states) {
    return {
        params[0] * states[0] - params[1] * states[0] * states[1],
        params[2] * states[0] * states[1] - params[3] * states[1]
    };
}

void solve_ode(double t0, double t_end, State& states, double h, const Param& params) {
    double t = t0;

    while (t < t_end) {
        if (t + h > t_end) {
            h = t_end - t;
        }

        // Runge-Kutta 4th order solver
        State k1 = ode_val(params, states);

        State tempstates = {states[0] + (h * k1[0]) / 2.0, states[1] + (h * k1[1]) / 2.0};
        State k2 = ode_val(params, tempstates);

        tempstates = {states[0] + (h * k2[0]) / 2.0, states[1] + (h * k2[1]) / 2.0};
        State k3 = ode_val(params, tempstates);

        tempstates = {states[0] + (h * k3[0]), states[1] + (h * k3[1])};
        State k4 = ode_val(params, tempstates);

        states[0] += (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * (h / 6.0);
        states[1] += (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * (h / 6.0);
        t += h;
    }
}