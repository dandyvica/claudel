#pragma once
#include <iostream>
#include <string>
#include <thread>

#include "cxxopts.hpp"

struct CliOptions {
  // input file to analyze
  std::string image;

  // buffer size used to look for patterns
  size_t buffer_size;

  // minimum file size to consider
  size_t min_size;

  // number of threads to use
  size_t nb_threads;

  // display progress bar
  bool progress_bar;

  // cons
  explicit CliOptions(int argc, char** argv);

  // print
  friend std::ostream& operator<<(std::ostream& os, const CliOptions& c);
};
