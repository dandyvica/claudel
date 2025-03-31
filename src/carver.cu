#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#define BLOCK_SIZE 256
#define JPEG_HEADER_1 0xFF
#define JPEG_HEADER_2 0xD8
#define JPEG_HEADER_3 0xFF

__global__ void searchJPEGHeaders(const unsigned char *data, size_t size, int *foundOffsets) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size - 2) {
    if (data[idx] == JPEG_HEADER_1 && data[idx + 1] == JPEG_HEADER_2 &&
        data[idx + 2] == JPEG_HEADER_3) {
      foundOffsets[idx] = 1; // Mark as a possible JPEG start
    }
  }
}

void carveFiles(const char *data, size_t size, size_t offset) {
  unsigned char *device_data; // this is located in GPU (aka device) RAM
  int *device_offsets;        // list of offsets found if a carved file is found

  // Allocate CUDA memory

  cudaMalloc((void **)&device_data, size);
  cudaMalloc((void **)&device_offsets, size * sizeof(int));
  cudaMemset(device_offsets, 0, size * sizeof(int));

  // copy from host
  cudaMemcpy(device_data, data, size, cudaMemcpyHostToDevice);

  // Launch CUDA Kernel
  int threadsPerBlock = BLOCK_SIZE;
  int numBlocks = (size + threadsPerBlock - 1) / threadsPerBlock;
  searchJPEGHeaders<<<numBlocks, threadsPerBlock>>>(device_data, size, device_offsets);

  // Copy results back
  std::vector<int> offsets(size);
  cudaMemcpy(offsets.data(), device_offsets, size * sizeof(int), cudaMemcpyDeviceToHost);

  // Extract identified JPEG files
  int fileCount = 0;
  for (size_t i = 0; i < size; i++) {
    if (offsets[i] == 1) {
      //   std::string outFilename = "carved_" + std::to_string(fileCount++) + ".jpg";
      //   std::ofstream outFile(outFilename, std::ios::binary);
      //   outFile.write(reinterpret_cast<char *>(&hostData[i]), 1024 * 1024); // Save up to 1MB
      //   outFile.close();
      //   std::cout << "Extracted: " << outFilename << " at offset " << i << std::endl;
      std::cout << "Extracted: at offset " << i+offset << std::endl;
    }
  }

  // Cleanup
  cudaFree(device_data);
  cudaFree(device_offsets);
}
