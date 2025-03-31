#include "args.hpp"

// read command line arguments
CliOptions::CliOptions(int argc, char **argv) {
    try {
        cxxopts::Options options("claudel", "yet another file carver");

        // Define available options
        options.add_options()("i,image", "Raw image file", cxxopts::value<std::string>()->default_value("image.raw"))(
            "b,bufsize", "Buffer size when reading image", cxxopts::value<size_t>()->default_value("4096"))(
            "m,minsize", "Minimum file size to consider", cxxopts::value<size_t>()->default_value("20000"))(
            "n,threads", "Number of threads to spawn", cxxopts::value<size_t>()->default_value("0"))(
            "g,gpu_specs", "Test for NVidia card and if found, print out GPU specs",
            cxxopts::value<bool>()->default_value("false"))("h,help", "Print help");

        // Parse arguments
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(0);
        }

        // Populate the structure
        image = result["image"].as<std::string>();
        buffer_size = result["bufsize"].as<size_t>();
        min_size = result["minsize"].as<size_t>();
        nb_threads = result["threads"].as<size_t>();
        gpu_specs = result["gpu_specs"].as<bool>();

        // by default, if not specified, we want the number of machine threads
        if (nb_threads == 0) {
            nb_threads = std::thread::hardware_concurrency();
        }

    } catch (const std::exception &e) {
        std::cerr << "Error: " << e.what() << std::endl;
        std::exit(1);
    }
}

// struct printing
std::ostream &operator<<(std::ostream &os, const CliOptions &c) {
    os << "CliOptions(";
    os << c.image << " ";
    os << c.buffer_size << " ";
    os << c.min_size << " ";
    os << c.nb_threads << ")";

    return os;
}