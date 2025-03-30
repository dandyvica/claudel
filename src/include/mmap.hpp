#pragma once

#include <boost/iostreams/device/mapped_file.hpp>
#include <fstream>
#include <iostream>

struct Mmap {
  // file mapped size
  std::size_t size;

  // mapped data in memory
  const char* data;

  // cons
  explicit Mmap(std::string& filepath);
};
