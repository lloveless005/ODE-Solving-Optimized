#ifndef WRITEFILE_H_
#define WRITEFILE_H_

#include <fstream>

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

#endif