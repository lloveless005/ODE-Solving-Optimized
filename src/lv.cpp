#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <chrono>
#include <omp.h>

using State = std::array<double, 2>;
using Param = std::array<double, 4>;

// Lotka-Volterra Model
// dx/dt = ax - bxy
// dy/dt = dxy - gy

// Parameters: a, b, d, g
// Variables: x, y

// Example parameters: {1.0, 0.1, 0.075, 1.5}
// Example initial conditions: {10, 5}

// 1. ODE

State ode_val(const Param& params, const State& states) {
    return {
        params[0] * states[0] - params[1] * states[0] * states[1],
        params[2] * states[0] * states[1] - params[3] * states[1]
    };
}

// 2. Runge-Kutta solver

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

// Main

// int main() {
//     constexpr Param params = {1.0, 0.1, 0.075, 1.5};
//     State states = {10.0, 5.0};

//     constexpr double t0 = 0.0;
//     constexpr double t_end = 10.0;
//     constexpr double h = 0.1;

//     solve_ode(t0, t_end, states, h, params);

//     std::cout << "End values\nx: " << std::fixed << std::setprecision(4) << states[0] << "\ny: " << states[1] << "\n";

//     return 0;
// }

int main() {
    auto start_total = std::chrono::steady_clock::now();
    constexpr size_t samples = 1 << 20;
    constexpr int numParams = 4;

    constexpr Param params = {1.0, 0.1, 0.075, 1.5};

    constexpr double t0 = 0.0;
    constexpr double t_end = 10.0;
    constexpr double h = 0.1;

    Param low_params, high_params;

    for (int i = 0; i < numParams; i++) {
        low_params[i] = params[i] * 0.85;
        high_params[i] = params[i] * 1.15;
    }

    // allocate memory for parameter sets
    std::vector<double> param_set(samples * numParams);

    // generate uniform random numbers between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    #pragma omp parallel for
    for (size_t i = 0; i < samples; i++) {
        for (int j = 0; j < numParams; j++) {
            param_set[i * numParams + j] = (0.85 + 0.3 * dist(gen)) * params[j];
        }
    }

    // solve odes for each parameter set
    std::vector<double> results(2 * samples);

    auto start_sim = std::chrono::steady_clock::now();
    #pragma omp parallel for
    for (size_t i = 0; i < samples; i++) {
        Param temp_params = {param_set[i * numParams], 
            param_set[i * numParams + 1], 
            param_set[i * numParams + 2],
            param_set[i * numParams + 3]
        };

        State temp_state = {10.0, 5.0};
        solve_ode(t0, t_end, temp_state, h, temp_params);

        results[i * 2] = temp_state[0];
        results[i * 2 + 1] = temp_state[1];
    }
    auto end_sim = std::chrono::steady_clock::now();
    auto elapsed_sim = std::chrono::duration_cast<std::chrono::milliseconds>(end_sim - start_sim);

    auto end_total = std::chrono::steady_clock::now();
    auto elapsed_total = std::chrono::duration_cast<std::chrono::milliseconds>(end_total - start_total);

    std::cout << "Simulation time: " << elapsed_sim.count() << " ms\n";
    std::cout << "Total time: " << elapsed_total.count() << " ms\n";


    return 0;
}