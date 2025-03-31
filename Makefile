# list of source files
SRC_FILES := $(wildcard src/*.cpp)

# list of object files
OBJ_FILES := $(subst .cpp,.o,$(SRC_FILES))
OBJ_FILES := $(subst src/,obj/,$(OBJ_FILES))

# executable to build
BIN = bin/claudel

# Compiler and flags
CXXFLAGS = -std=c++23 -fmodules-ts -I ./src/include -I /usr/local/cuda-12.8/targets/x86_64-linux/include -O2 -s
LDFLAGS = -L/usr/local/cuda-12.8/targets/x86_64-linux/lib -lboost_iostreams -lboost_system -lcuda -lcudart

# rule for binary
$(BIN): $(OBJ_FILES) obj/carver.o
	g++-13 $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

# rules for objs
obj/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# rules for Cuda
obj/carver.o: src/carver.cu
	nvcc -O2 -c $< -o $@


# # Print variables using warning
# print-vars:
# 	$(warning SRC_FILES = $(SRC_FILES))
# 	$(warning OBJ_FILES = $(OBJ_FILES))

# # Default target to show the print-vars
# all: print-vars

# # Clean up the generated files
clean:
	rm -f $(OBJ_FILES) $(BIN)

.PHONY: clean run
