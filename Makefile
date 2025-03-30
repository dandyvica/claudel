# Makefile for building C++20 modules with g++-13

# Compiler and flags
CXXFLAGS = -std=c++23 -fmodules-ts -I ./src/include

BIN = bin/claudel
SRC = $(wildcard src/*.cpp)      # All C++ source files in the src directory

# Build object files
obj/args.o: src/args.cpp
	g++-13 $(CXXFLAGS) -c $< -o $@

obj/mmap.o: src/mmap.cpp
	g++-13 $(CXXFLAGS) -c $< -o $@

obj/claudel.o: src/claudel.cpp
	g++-13 $(CXXFLAGS) -c $< -o $@

obj/gpu.o: src/gpu.cpp
	g++-13 $(CXXFLAGS) -c $< -o $@

# CUDA
# obj/gpu.o: src/gpu.cu
# 	nvcc -O2 -c $< -o $@

# compile and link
OBJ =  obj/args.o obj/mmap.o obj/claudel.o obj/gpu.o
$(BIN): $(OBJ)
	g++-13 $(CXXFLAGS) $(OBJ) -L/usr/local/cuda-12.8/targets/x86_64-linux/lib -o $@ -lboost_iostreams -lboost_system -lcuda -lcudart

# # Clean up the generated files
clean:
	rm -f $(OBJ) $(BIN)

# # Run the executable
# run: $(EXECUTABLE)
# 	./$(EXECUTABLE)

.PHONY: clean run
