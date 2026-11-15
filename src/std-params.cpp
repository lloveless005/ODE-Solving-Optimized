#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <format>
#include <chrono>
#include <vector>
#include <algorithm>
#include <globals.h>

#define FILE_NAME_FP64 "input/standard-parameters-fp64.bin"
#define FILE_NAME_FP32 "input/standard-parameters-fp32.bin"

using namespace std;

template <typename T>
bool WriteFile(ofstream& outFile, vector<T>& inList) {
    constexpr size_t chunk_size = 64*1024*1024;
    size_t total_bytes = inList.size() * sizeof(T);
    size_t written = 0;

    while (written < total_bytes) {
        size_t to_write = min(chunk_size, total_bytes - written);
        outFile.write(reinterpret_cast<const char*>(&inList[written / sizeof(T)]), to_write);
        if (!outFile) return false;
        written += to_write;
    }

    return true;
}

int main() {
    auto start_total = chrono::steady_clock::now();
    int error_status = 0;

    // initialize output file stream
    ofstream outputFile64;
    outputFile64.open(FILE_NAME_FP64, ios::binary);

    ofstream outputFile32;
    outputFile32.open(FILE_NAME_FP32, ios::binary);

    vector<double> param_set_fp64(SAMPLES * NUM_PARAMS);
    constexpr double default_params_fp64[] = {1.0, 0.1, 0.075, 1.5};

    vector<float> param_set_fp32(SAMPLES * NUM_PARAMS);


    if (!outputFile64.is_open()) {

        cout << "Could not open " << FILE_NAME_FP64 << endl;
        error_status = 1;

    } else if (!outputFile32.is_open()) {

        cout << "Could not open " << FILE_NAME_FP32 << endl;
        error_status = 2;

    } else {

        // generate random parameters
        auto start_gen = chrono::steady_clock::now();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            mt19937 gen(1337 + tid);
            uniform_real_distribution<double> dist(0.0, 1.0);

            #pragma omp for
            for (size_t i = 0; i < SAMPLES; i++) {
                for (int j = 0; j < NUM_PARAMS; j++) {
                    param_set_fp64[i * NUM_PARAMS + j] = (0.85 + 0.3 * dist(gen)) * default_params_fp64[j];
                }
            }
        }
        auto end_gen = chrono::steady_clock::now();
        auto elapsed_gen = chrono::duration_cast<chrono::milliseconds>(end_gen - start_gen);
        cout << "Parameter generation time: " << elapsed_gen.count() << " ms" << endl;


        auto start_write = chrono::steady_clock::now();

        // write FP64 parameter set as binary to output file
        // outputFile64.write(reinterpret_cast<const char*>(param_set_fp64.data()), param_set_fp64.size() * sizeof(double));
        WriteFile(outputFile64, param_set_fp64);

        #pragma omp parallel for 
        for (int idx = 0; idx < SAMPLES * NUM_PARAMS; idx++) {
            param_set_fp32[idx] = static_cast<float>(param_set_fp64[idx]);
        }

        // outputFile32.write(reinterpret_cast<const char*>(param_set_fp32.data()), param_set_fp32.size() * sizeof(float));
        WriteFile(outputFile32, param_set_fp32);

        auto end_write = chrono::steady_clock::now();
        auto elapsed_write = chrono::duration_cast<chrono::milliseconds>(end_write - start_write);
        cout << "File write time: " << elapsed_write.count() << " ms" << endl;
    }

    outputFile64.close();
    outputFile32.close();

    auto end_total = chrono::steady_clock::now();
    auto elapsed_total = chrono::duration_cast<chrono::milliseconds>(end_total - start_total);
    cout << "Total time: " << elapsed_total.count() << " ms" << endl;

    return error_status;
}