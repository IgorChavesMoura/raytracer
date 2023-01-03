//
// Created by moura on 30/12/2022.
//

#include <iostream>

using namespace std;

#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
                  file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

__global__ void add(int a, int b, int* c) {
    *c = a + b;
}

int main() {
    int count;
    checkCudaErrors(cudaGetDeviceCount(&count));

    cout << "Found " << count << " device(s)" << endl;

    for (int i = 0; i < count; ++i) {
        cudaDeviceProp prop;

        checkCudaErrors(cudaGetDeviceProperties(&prop, i));
        cout << "Device name: " << prop.name << endl;
        cout << "Total Memory: " << prop.totalGlobalMem / 1024.0 / 1024.0 << "MB" << endl;
        cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << endl;
        cout << "Compute capability: " << prop.major << "." << prop.minor << endl;
    }

    int result;
    int* devResult;

    checkCudaErrors(cudaMalloc((void**)&devResult, sizeof(int)));

    add<<<1, 1>>>(7, 8, devResult);
    //checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(&result, devResult, sizeof(int), cudaMemcpyDeviceToHost));

    cout << "7 + 8 = " << result << endl;

    cudaFree(devResult);

    return 0;
}