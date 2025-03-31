#include <algorithm>
#include <iostream>

#include "args.hpp"
#include "mmap.hpp"
// #include "gpu.hpp"
#include "aho_corasick.hpp"
#include "corpus.hpp"

using namespace std;

// extern funcs
void carveFiles(const char *data, size_t size, size_t offset);
int gpu_specs();

int main(int argc, char **argv) {
    // get cli arguments
    CliOptions opts(argc, argv);

    // get GPU capabilities if requested
    if (opts.gpu_specs) {
      gpu_specs();
      return 1;
    }
    // auto rc = gpu_specs();

    // Open image file and mapped it in memory
    Mmap mmap(opts.image);
    cout << "file size: " << mmap.size << endl;

    // // we're going to process the file concurrently
    // size_t chunk_size = mmap.size / opts.nb_threads;

    // // we'll our threads IDs here
    // vector<thread> threads;

    // for (int i = 0; i < opts.nb_threads; ++i) {
    //   // these are indexes of mmaped data. Each thread have its own chunk of data
    //   size_t start = i * chunk_size;
    //   size_t end = (i == opts.nb_threads - 1) ? mmap.size : (i + 1) * chunk_size;

    //   //threads.emplace_back(read_data, std::ref(file_map), start, end, i);
    // }

    // // Join all threads
    // for (auto& t : threads) {
    //   t.join();
    // }

    size_t chunk_size = 100000000;
    for (int i = 0; i < mmap.size / chunk_size; ++i) {
        // these are indexes of mmaped data. Each thread have its own chunk of data
        size_t start = i * chunk_size;
        size_t end = std::min((i + 1) * chunk_size, mmap.size);

        // cout << start << " " << end << endl;
        carveFiles(mmap.data + start, chunk_size, start);
    }

    return 0;
}

// give the lower and upper bounds for a chunk
// std::tuple<size_t, size_t> bounds(size_t file_size, size_t chunk_size) {
//   size_t chunk_length = file_size / chunk_size;
// }