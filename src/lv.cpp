#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <chrono>
#include <omp.h>
#include <fstream>
#include <vector>
#include <cmath>
#include <globals.h>
#include <writefile.h>
#include <string>

// Lotka-Volterra Model
// dx/dt = ax - bxy
// dy/dt = cxy - dy

// Parameters: a, b, c, d
// Variables: x, y

// Example parameters: {1.0, 0.1, 0.075, 1.5}
// Example initial conditions: {10, 5}

using namespace std;

using State = array<double, 2>;
using Param = array<double, 4>;

#define INPUT_FILE_FP64 "input/standard-parameters-fp64.bin"
const string OUTPUT_FILE_FP64 = "output/cpu-64-" + to_string(static_cast<int>(log2(SAMPLES))) + ".bin";


State ode_val(const Param& params, const State& in_state) {
    return {
        params[0] * in_state[0] - params[1] * in_state[0] * in_state[1],
        params[2] * in_state[0] * in_state[1] - params[3] * in_state[1]
    };
}

void solve_ode(int steps, State& in_state, double h, const Param& params) {
    int current = 0;

    while (current < steps) {

        // Runge-Kutta 4th order solver
        State k1 = ode_val(params, in_state);
        State temp_states = {in_state[0] + (h * k1[0]) / 2.0, in_state[1] + (h * k1[1]) / 2.0};

        State k2 = ode_val(params, temp_states);
        temp_states = {in_state[0] + (h * k2[0]) / 2.0, in_state[1] + (h * k2[1]) / 2.0};

        State k3 = ode_val(params, temp_states);
        temp_states = {in_state[0] + (h * k3[0]), in_state[1] + (h * k3[1])};

        State k4 = ode_val(params, temp_states);

        in_state[0] += (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) * (h / 6.0);
        in_state[1] += (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) * (h / 6.0);
        current++;
    }
}


int main() {
    int error_status = 0;

    auto start_total = chrono::steady_clock::now();

    constexpr double t0 = 0.0;
    constexpr double t_end = 10.0;
    constexpr double h = 0.1;
    constexpr int steps = ceil((t_end - t0) / h);

    // Fill param_set from file

    vector<double> param_set_fp64(SAMPLES * NUM_PARAMS);

    ifstream inputFile64;
    inputFile64.open(INPUT_FILE_FP64);

    if (!inputFile64.is_open()) {
        cout << "Failed to open " << INPUT_FILE_FP64 << endl;
        error_status = 1;
    } else {
        inputFile64.read(reinterpret_cast<char*>(param_set_fp64.data()), SAMPLES * NUM_PARAMS * sizeof(double));
    }


    // Output benchmark for parameter reading/generation

    auto end_params = chrono::steady_clock::now();
    auto elapsed_params = chrono::duration_cast<chrono::milliseconds>(end_params - start_total);

    // Solve ODE system for each parameter set

    vector<double> results_fp64(2 * SAMPLES);

    auto start_sim_fp64 = chrono::steady_clock::now();

    // FP64
    #pragma omp parallel for
    for (size_t i = 0; i < SAMPLES; i++) {
        Param temp_params = {param_set_fp64[i * NUM_PARAMS], 
            param_set_fp64[i * NUM_PARAMS + 1], 
            param_set_fp64[i * NUM_PARAMS + 2],
            param_set_fp64[i * NUM_PARAMS + 3]
        };

        State temp_state = {10.0, 5.0};
        solve_ode(steps, temp_state, h, temp_params);

        results_fp64[i * 2] = temp_state[0];
        results_fp64[i * 2 + 1] = temp_state[1];
    }
    auto end_sim_fp64 = chrono::steady_clock::now();
    auto elapsed_sim_fp64 = chrono::duration_cast<chrono::milliseconds>(end_sim_fp64 - start_sim_fp64);

    cout << endl << "FP64" << endl << "---------------------------------" << endl;

    for (int idx = 0; idx < 5; idx++) {
        cout << "State " << idx << ": " << results_fp64[idx * 2] << " " << results_fp64[idx * 2 + 1] << endl;
    }

    auto start_write = chrono::steady_clock::now();

    ofstream outputFile64;
    outputFile64.open(OUTPUT_FILE_FP64, ios::binary);

    if(!outputFile64.is_open()) {
        cout << "Failed to open " << OUTPUT_FILE_FP64 << endl;
        error_status = 2;
    } else {
        WriteFile(outputFile64, results_fp64);
    }

    outputFile64.close();

    auto end_write = chrono::steady_clock::now();
    auto elapsed_write = chrono::duration_cast<chrono::milliseconds>(end_write - start_write);

    cout << endl << "Parameter read time: " << elapsed_params.count() << " ms\n";
    cout << "Simulation time: " << elapsed_sim_fp64.count() << " ms\n";
    cout << "Write time: " << elapsed_write.count() << " ms" << endl;

    auto end_total = chrono::steady_clock::now();
    auto elapsed_total = chrono::duration_cast<chrono::milliseconds>(end_total - start_total);

    cout << endl << "Total time: " << elapsed_total.count() << " ms" << endl;

    return error_status;
}