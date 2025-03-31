#pragma once
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>

/**
 * @struct FileType
 * @brief Store the different features of a file type
 */
struct FileType {
  std::unique_ptr<std::byte[]> magic; /**< the magic bytes to look for */
  std::string ext;                    /**< the file type extension */

  // the function used to carve
  // pub carving_func: CarvingFunc,

  std::string category; /**< category like images, audio, etc */
  size_t min_size;      /**< the minimum file size to consider for this file type */
  size_t index;         /**< the current index of the file being carved */

  /**
   * @brief Once we have carved and found a potential file, we have to save it
   * on disk
   * @param size The payload (bytes buffer) to write on disk
   */
  int save_file(FileType &self, std::span<uint8_t> payload);
};

/**
 * @struct Corpus
 * @brief A corpus is a list of file types to look for
 */
struct Corpus {
  std::unordered_map<int, std::string> map; /**< this map holds the list of file types */

  explicit Corpus();
};
