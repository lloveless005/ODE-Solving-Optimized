.PHONY: all clean gpu

CXX 	    = g++
LD 	        = g++
NVCC        = nvcc
CXXFLAGS    = -O3 -march=native -Wall -Wextra -fopenmp
NVCCFLAGS   = -O3 -arch=native
LDFLAGS     = -fopenmp -ggdb
SRC_DIR     = src
OBJ_DIR     = obj
BIN_DIR     = build
TARGET 	    = $(BIN_DIR)/lv
TARGET_NF   = $(BIN_DIR)/lv-nf
TARGET_GPU  = $(BIN_DIR)/lv-cu
TARGET_STDP = $(BIN_DIR)/std-params

all: $(TARGET)

$(TARGET): $(OBJ_DIR)/lv.o | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/lv.o: $(SRC_DIR)/lv.cpp | $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

noflags: $(TARGET_NF)

$(TARGET_NF): $(OBJ_DIR)/lv-nf.o | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/lv-nf.o: $(SRC_DIR)/lv.cpp | $(OBJ_DIR)
	$(CXX) -c -Wall -Wextra -fopenmp $< -o $@

gpu: $(TARGET_GPU)

$(TARGET_GPU): $(OBJ_DIR)/lv-gpu.o | $(BIN_DIR)
	$(NVCC) -lcurand $< -o $@

$(OBJ_DIR)/lv-gpu.o: $(SRC_DIR)/lv-gpu.cu | $(OBJ_DIR)
	$(NVCC) -c $(NVCCFLAGS) $< -o $@

std-params: $(TARGET_STDP)

$(TARGET_STDP): $(OBJ_DIR)/std-params.o | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/std-params.o: $(SRC_DIR)/std-params.cpp | $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@