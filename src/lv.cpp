#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <chrono>
#include <omp.h>
#include <fstream>

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

#define NUM_PARAMS 4
#define SAMPLES (1 << 20)

#define INPUT_FILE_FP64 "input/standard-parameters-fp64.txt"
#define INPUT_FILE_FP32 "input/standard-parameters-fp32.txt"
constexpr bool use_input_file = true;


State ode_val(const Param& params, const State& in_state);

void solve_ode(double t0, double t_end, State& in_state, double h, const Param& params);


int main() {
    int error_status = 0;

    auto start_total = chrono::steady_clock::now();

    constexpr Param default_params = {1.0, 0.1, 0.075, 1.5};

    constexpr double t0 = 0.0;
    constexpr double t_end = 10.0;
    constexpr double h = 0.1;

    // Generate 2**20 random parameters within +-15% of the default values 
    vector<double> param_set_fp64(SAMPLES * NUM_PARAMS);
    vector<float> param_set_fp32(SAMPLES * NUM_PARAMS);

    if (use_input_file) {

        // Fill param_set from file

        ifstream inputFile64;
        inputFile64.open(INPUT_FILE_FP64);

        ifstream inputFile32;
        inputFile32.open(INPUT_FILE_FP32);

        if (!inputFile64.is_open()) {
            cout << "Failed to open " << INPUT_FILE_FP64 << endl;
            error_status = 1;
        } else if (!inputFile32.is_open()) {
            cout << "Failed to open " << INPUT_FILE_FP32 << endl;
            error_status = 2;
        } else {
            inputFile64.read(reinterpret_cast<char*>(param_set_fp64.data()), SAMPLES * NUM_PARAMS * sizeof(double));
            inputFile32.read(reinterpret_cast<char*>(param_set_fp32.data()), SAMPLES * NUM_PARAMS * sizeof(float));
        }

    } else {

        // Generate random parameters if there is no file

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            mt19937 gen(1337 + tid);
            uniform_real_distribution<double> dist(0.0, 1.0);

            #pragma omp for
            for (size_t i = 0; i < SAMPLES; i++) {
                for (int j = 0; j < NUM_PARAMS; j++) {
                    param_set_fp64[i * NUM_PARAMS + j] = (0.85 + 0.3 * dist(gen)) * default_params[j];
                    param_set_fp32[i * NUM_PARAMS + j] = static_cast<float>(param_set_fp64[i * NUM_PARAMS + j]);
                }
            }
        }

    }

    // Output benchmark for parameter reading/generation

    auto end_params = chrono::steady_clock::now();
    auto elapsed_params = chrono::duration_cast<chrono::milliseconds>(end_params - start_total);

    if (use_input_file) cout << "Parameter read time: " << elapsed_params.count() << " ms\n";
    else cout << "Parameter generation time: " << elapsed_params.count() << " ms\n";

    // Solve ODE system for each parameter set

    auto start_sim_fp64 = chrono::steady_clock::now();

    vector<double> results_fp64(2 * SAMPLES);

    #pragma omp parallel for
    for (size_t i = 0; i < SAMPLES; i++) {
        Param temp_params = {param_set_fp64[i * NUM_PARAMS], 
            param_set_fp64[i * NUM_PARAMS + 1], 
            param_set_fp64[i * NUM_PARAMS + 2],
            param_set_fp64[i * NUM_PARAMS + 3]
        };

        State temp_state = {10.0, 5.0};
        solve_ode(t0, t_end, temp_state, h, temp_params);

        results_fp64[i * 2] = temp_state[0];
        results_fp64[i * 2 + 1] = temp_state[1];
    }
    auto end_sim_fp64 = chrono::steady_clock::now();
    auto elapsed_sim_fp64 = chrono::duration_cast<chrono::milliseconds>(end_sim_fp64 - start_sim_fp64);

    cout << endl << "FP64" << endl << "---------------------------------" << endl;
    cout << "Simulation time: " << elapsed_sim_fp64.count() << " ms\n";

    for (int idx = 0; idx < 5; idx++) {
        cout << "State " << idx << ": " << results_fp64[idx * 2] << " " << results_fp64[idx * 2 + 1] << endl;
    }

    cout << endl << "FP32" << endl << "---------------------------------" << endl;

    auto end_total = chrono::steady_clock::now();
    auto elapsed_total = chrono::duration_cast<chrono::milliseconds>(end_total - start_total);

    cout << endl << "Total time: " << elapsed_total.count() << " ms\n";

    return error_status;
}


State ode_val(const Param& params, const State& in_state) {
    return {
        params[0] * in_state[0] - params[1] * in_state[0] * in_state[1],
        params[2] * in_state[0] * in_state[1] - params[3] * in_state[1]
    };
}

void solve_ode(double t0, double t_end, State& in_state, double h, const Param& params) {
    double t = t0;

    while (t < t_end) {
        if (t + h > t_end) {
            h = t_end - t;
        }

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
        t += h;
    }
}