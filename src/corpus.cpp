#include "corpus.hpp"

int FileType::save_file(FileType &self, std::span<uint8_t> payload) {
  std::mutex mtx; // mutex to protect index

  // create directory if not yet existing
  if (!std::filesystem::exists(self.category)) {
    // Attempt to create the directory and its parent directories
    if (!std::filesystem::create_directories(self.category)) {
      std::cout << "failed to create " << self.category << " directory" << std::endl;
      return -1;
    }
  }

  // now we can build file name
  std::ostringstream file_name;
  size_t index{0};

  {
    std::lock_guard<std::mutex> lock(mtx);
    index = self.index;
  }

  file_name << self.category << "/" << self.ext << std::setfill('0') << std::setw(8) << index << "."
            << self.ext;

  // now write file to disk
  std::ofstream file(file_name.str(), std::ios::binary); // Open file in binary mode
  if (!file) {
    std::cerr << "error opening file " << file_name.str() << " for writing" << std::endl;
    return -1;
  }

  file.write(reinterpret_cast<const char *>(payload.data()), payload.size());
  file.flush();
  file.close();

  // add 1 to our counter
  {
    std::lock_guard<std::mutex> lock(mtx);
    self.index++;
  }

  return 0;
}

Corpus::Corpus() {
  std::unordered_map<int, FileType> map;

  // BITMAP
  auto magic = std::make_unique<std::byte[]>(2);
  magic[0] = std::byte{'B'};
  magic[1] = std::byte{'M'};

  map[0] = FileType{
    magic : std::move(magic),
    ext : std::string("bmp"),
    category : std::string("images/bmp"),
    min_size : 10000,
    index : 0,
  };
}
