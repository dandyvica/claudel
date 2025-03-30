#include "mmap.hpp"

namespace io = boost::iostreams;

// load file int RAM using mmap
Mmap::Mmap(std::string& filename) {
  try {
    // Open memory-mapped file in read-only mode
    io::mapped_file_source mmap(filename);

    if (!mmap.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        std::exit(2);
    }    

    size = mmap.size();
    data = mmap.data();

  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << std::endl;
    std::exit(2);
  }
}
