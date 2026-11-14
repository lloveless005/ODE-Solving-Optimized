#include <iostream>
#include <fstream>
#include <omp.h>
#include <random>
#include <format>
#include <chrono>

#define FILE_NAME "input/standard-parameters.txt"
#define NUM_PARAMS 4
constexpr int SAMPLES = 1 << 20;

using namespace std;

int main() {
    auto start_total = chrono::steady_clock::now();
    int error_status = 0;

    // initialize output file stream
    ofstream outputFile;
    outputFile.open(FILE_NAME);

    double* param_set= new double[SAMPLES * NUM_PARAMS];
    constexpr double default_params[] = {1.0, 0.1, 0.075, 1.5};

    if (!outputFile.is_open()) {

        cout << "Could not open " << FILE_NAME << endl;
        error_status = 1;

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
                    param_set[i * NUM_PARAMS + j] = (0.85 + 0.3 * dist(gen)) * default_params[j];
                }
            }
        }
        auto end_gen = chrono::steady_clock::now();
        auto elapsed_gen = chrono::duration_cast<chrono::milliseconds>(end_gen - start_gen);
        cout << "Parameter generation time: " << elapsed_gen.count() << " ms" << endl;


        auto start_write = chrono::steady_clock::now();

        for (int idx = 0; idx < SAMPLES * NUM_PARAMS; idx++) {
            outputFile << format("{}", param_set[idx]) << endl;
        }

        auto end_write = chrono::steady_clock::now();
        auto elapsed_write = chrono::duration_cast<chrono::milliseconds>(end_write - start_write);
        cout << "Parameter generation time: " << elapsed_write.count() << " ms" << endl;
    }

    delete [] param_set;
    outputFile.close();

    auto end_total = chrono::steady_clock::now();
    auto elapsed_total = chrono::duration_cast<chrono::milliseconds>(end_total - start_total);
    cout << "Total time: " << elapsed_total.count() << " ms" << endl;

    return error_status;
}