#include <fstream>
#include <string>
#include <globals.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <numeric>
#include <chrono>

using namespace std;

const string INPUT_FILE_CPU_FP64 = "output/cpu-64-" + to_string(static_cast<int>(log2(SAMPLES))) + ".bin";
const string INPUT_FILE_GPU_FP64 = "output/gpu-64-" + to_string(static_cast<int>(log2(SAMPLES))) + ".bin";
const string INPUT_FILE_GPU_FP32 = "output/gpu-32-" + to_string(static_cast<int>(log2(SAMPLES))) + ".bin";

int main() {
    int error_status = 0;

    ifstream inputCPU64, inputGPU64, inputGPU32;

    inputCPU64.open(INPUT_FILE_CPU_FP64, ios::binary);
    inputGPU64.open(INPUT_FILE_GPU_FP64, ios::binary);
    inputGPU32.open(INPUT_FILE_GPU_FP32, ios::binary);

    if (!inputCPU64.is_open()) {

        cout << "Failed to open " << INPUT_FILE_CPU_FP64 << endl;
        error_status = 1;

    } else if (!inputGPU64.is_open()) {

        cout << "Failed to open " << INPUT_FILE_GPU_FP64 << endl;
        error_status = 2;

    } else if (!inputGPU32.is_open()) {

        cout << "Failed to open " << INPUT_FILE_GPU_FP32 << endl;
        error_status = 3;

    } else {

        vector<double> state_cpu64(2 * SAMPLES);
        vector<double> state_gpu64(2 * SAMPLES);
        vector<float> state_gpu32(2 * SAMPLES);

        vector<double> cpu_gpu_diff(2 * SAMPLES);
        vector<double> precision_diff(2 * SAMPLES);

        // Load files into vectors

        inputCPU64.read(reinterpret_cast<char*>(state_cpu64.data()), 2 * SAMPLES * sizeof(double));
        inputGPU64.read(reinterpret_cast<char*>(state_gpu64.data()), 2 * SAMPLES * sizeof(double));
        inputGPU32.read(reinterpret_cast<char*>(state_gpu32.data()), 2 * SAMPLES * sizeof(float));

        // Comparisons

        #pragma omp parallel for
        for (size_t idx = 0; idx < 2 * SAMPLES; idx++) {
            cpu_gpu_diff[idx] = abs(state_cpu64[idx] - state_gpu64[idx]);
        }

        #pragma omp parallel for
        for (size_t idx = 0; idx < 2 * SAMPLES; idx++) {
            precision_diff[idx] = abs(state_gpu64[idx] - static_cast<double>(state_gpu32[idx]));
        }

        double max_cpu_gpu_diff, max_precision_diff;

        max_cpu_gpu_diff = *max_element(cpu_gpu_diff.begin(), cpu_gpu_diff.end());
        max_precision_diff = *max_element(precision_diff.begin(), precision_diff.end());

        cout << endl << "Max absolute difference between CPU FP64 and GPU FP64: " << max_cpu_gpu_diff << endl;
        cout << "Max absolute difference between GPU FP64 and GPU FP32: " << max_precision_diff << endl;

        // Relative difference

        #pragma omp parallel for
        for (size_t idx = 0; idx < 2 * SAMPLES; idx++) {
            cpu_gpu_diff[idx] /= abs(state_cpu64[idx]);
        }

        #pragma omp parallel for
        for (size_t idx = 0; idx < 2 * SAMPLES; idx++) {
            precision_diff[idx] /= abs(state_gpu64[idx]);
        }

        max_cpu_gpu_diff = *max_element(cpu_gpu_diff.begin(), cpu_gpu_diff.end());
        max_precision_diff = *max_element(precision_diff.begin(), precision_diff.end());

        cout << endl << "Max relative difference between CPU FP64 and GPU FP64: " << max_cpu_gpu_diff << endl;
        cout << "Max relative difference between GPU FP64 and GPU FP32: " << max_precision_diff << endl;

        // Average difference

        auto avg_start = chrono::steady_clock::now();

        double cg_sum = 0.0;
        double p_sum = 0.0;

        #pragma omp parallel for reduction(+:cg_sum)
        for (size_t idx = 0; idx < cpu_gpu_diff.size(); idx++) cg_sum += cpu_gpu_diff[idx];

        double cg_avg = cg_sum / cpu_gpu_diff.size();

        #pragma omp parallel for reduction(+:p_sum)
        for (size_t idx = 0; idx < precision_diff.size(); idx++) p_sum += precision_diff[idx];

        double p_avg = p_sum / precision_diff.size();

        auto avg_end = chrono::steady_clock::now();
        auto avg_elapsed = chrono::duration_cast<chrono::milliseconds>(avg_end - avg_start);

        cout << endl << "Sum and avg time: " << avg_elapsed << endl;

        cout << endl << "Average relative difference between CPU FP64 and GPU FP64: " << cg_avg << endl;
        cout << "Average relative difference between GPU FP64 and GPU FP32: " << p_avg << endl;


    }

    return error_status;
}