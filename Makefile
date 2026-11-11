.PHONY: all clean gpu

CXX 	    = g++
LD 	        = g++
NVCC        = nvcc
CXXFLAGS    = -O3 -march=native -Wall -Wextra -fopenmp
NVCCFLAGS   = -O3 -arch=sm_120
LDFLAGS     = -fopenmp
SRC_DIR     = src
OBJ_DIR     = obj
BIN_DIR     = build
OBJS        = $(OBJ_DIR)/lv.o
TARGET 	    = $(BIN_DIR)/lv
TARGET_GPU  = $(BIN_DIR)/lv-cu

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

gpu: $(TARGET_GPU)

$(TARGET_GPU): $(OBJ_DIR)/lv-gpu.o | $(OBJ_DIR)
	$(NVCC) $< -o $@

$(OBJ_DIR)/lv-gpu.o: $(SRC_DIR)/lv-gpu.cu
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@