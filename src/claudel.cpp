#include <iostream>

#include "args.hpp"
#include "mmap.hpp"
#include "gpu.hpp"

using namespace std;

int main(int argc, char** argv) {
  // get cli arguments
  CliOptions opts(argc, argv);

  // get GPU capabilities
  auto rc = gpu_specs();

  // Open image file and mapped it in memory
  Mmap mmap(opts.image);
  cout << "file size: " << mmap.size << endl;

  // we're going to process the file concurrently
  size_t chunk_size = mmap.size / opts.nb_threads;

  // we'll our threads IDs here
  vector<thread> threads;

  for (int i = 0; i < opts.nb_threads; ++i) {
    // these are indexes of mmaped data. Each thread have its own chunk of data
    size_t start = i * chunk_size;
    size_t end = (i == opts.nb_threads - 1) ? mmap.size : (i + 1) * chunk_size;

    //threads.emplace_back(read_data, std::ref(file_map), start, end, i);
  }

  // Join all threads
  for (auto& t : threads) {
    t.join();
  }

  return 0;
}