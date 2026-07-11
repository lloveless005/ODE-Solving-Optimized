#include <iostream>
#include <iomanip>
#include <array>
#include <random>
#include <chrono>

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
//     constexpr double h = 0.001;

//     solve_ode(t0, t_end, states, h, params);

//     std::cout << "End values\nx: " << std::fixed << std::setprecision(2) << states[0] << "\ny: " << states[1] << "\n";

//     return 0;
// }

int main() {
    constexpr size_t samples = 1 << 12;
    constexpr int numParams = 4;

    constexpr Param params = {1.0, 0.1, 0.075, 1.5};

    constexpr double t0 = 0.0;
    constexpr double t_end = 10.0;
    constexpr double h = 0.001;

    Param low_params, high_params;

    for (int i = 0; i < numParams; i++) {
        low_params[i] = params[i] * 0.85;
        high_params[i] = params[i] * 1.15;
    }

    // allocate memory for parameter sets
    std::array<double, samples * numParams> param_set;

    // generate uniform random numbers between 0 and 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (size_t i = 0; i < samples * numParams; i++) {
        param_set[i] = dist(gen);
    }

    // scale uniform numbers to parameters
    for (int i = 0; i < numParams; i++) {
        for (size_t j = 0; j < samples; j++) {
            param_set[i + j * numParams] = (0.85 + 0.3 * param_set[i + j * numParams]) * params[i];
        }
    }

    // solve odes for each parameter set
    std::array<double, 2 * samples> results;

    auto start = std::chrono::steady_clock::now();
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
    auto end = std::chrono::steady_clock::now();
    auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // output
    for (int i = 0; i < 20; i += 2) {
        std::cout << "Result " << i / 2 << "\nx: " << std::fixed << std::setprecision(2)
                  << results[i] << "\ny: " << results[i + 1] << "\n";
    }

    std::cout << "Simulation time: " << elapsed_ms.count() << " ms\n";
    std::cout << "Simulations per second: " << static_cast<int>(samples / static_cast<double>(elapsed_ms.count()) * 1000.0) << "\n";


    return 0;
}