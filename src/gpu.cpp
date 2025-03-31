#include <cuda_runtime.h>
#include <iostream>

int gpu_specs() {
    int deviceCount = 0;

    // Step 1: Get the number of CUDA-capable devices
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cout << "No CUDA-capable devices found." << std::endl;
        return 0;
    }

    // Step 2: Iterate through the devices and check if any are NVIDIA GPUs
    bool foundNvidiaGPU = false;
    for (int i = 0; i < deviceCount; ++i) {
        int nDevices;
        cudaGetDeviceCount(&nDevices);

        printf("Number of devices: %d\n", nDevices);

        for (int i = 0; i < nDevices; i++) {
            cudaDeviceProp prop;
            cudaGetDeviceProperties(&prop, i);
            printf("Device Number: %d\n", i);
            printf("  Device name: %s\n", prop.name);
            printf("  Memory Clock Rate (MHz): %d\n", prop.memoryClockRate / 1024);
            printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
            printf("  Peak Memory Bandwidth (GB/s): %.1f\n",
                   2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
            printf("  Total global memory (Gbytes) %.1f\n",
                   (float)(prop.totalGlobalMem) / 1024.0 / 1024.0 / 1024.0);
            printf("  Shared memory per block (Kbytes) %.1f\n",
                   (float)(prop.sharedMemPerBlock) / 1024.0);
            printf("  minor-major: %d-%d\n", prop.minor, prop.major);
            printf("  Warp-size: %d\n", prop.warpSize);
            printf("  Concurrent kernels: %s\n", prop.concurrentKernels ? "yes" : "no");
            printf("  Concurrent computation/communication: %s\n\n",
                   prop.deviceOverlap ? "yes" : "no");
        }
    }

    if (!foundNvidiaGPU) {
        std::cout << "No NVIDIA GPU found." << std::endl;
    }

    return 0;
}