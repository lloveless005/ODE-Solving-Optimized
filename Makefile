.PHONY: all clean gpu noflags std-params

CXX 	    = g++
LD 	        = g++
NVCC        = nvcc
CXXFLAGS    = -O3  -I$(INC_DIR) -march=native -Wall -Wextra -fopenmp -ggdb
NVCCFLAGS   = -O3 -arch=native
LDFLAGS     = -fopenmp
SRC_DIR     = ./src
INC_DIR     = ./inc
OBJ_DIR     = ./obj
BIN_DIR     = ./build
TARGET 	    = $(BIN_DIR)/lv
TARGET_NF   = $(BIN_DIR)/lv-nf
TARGET_GPU  = $(BIN_DIR)/lv-cu
TARGET_STDP = $(BIN_DIR)/std-params

all: $(TARGET) $(TARGET_NF) $(TARGET_GPU) $(TARGET_STDP)

$(TARGET): $(OBJ_DIR)/lv.o | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/lv.o: $(SRC_DIR)/lv.cpp $(INC_DIR)/globals.h | $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

noflags: $(TARGET_NF)

$(TARGET_NF): $(OBJ_DIR)/lv-nf.o | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/lv-nf.o: $(SRC_DIR)/lv.cpp $(INC_DIR)/globals.h | $(OBJ_DIR)
	$(CXX) -c -I$(INC_DIR) -Wall -Wextra -fopenmp $< -o $@

gpu: $(TARGET_GPU)

$(TARGET_GPU): $(OBJ_DIR)/lv-gpu.o | $(BIN_DIR)
	$(NVCC) -lcurand $^ -o $@

$(OBJ_DIR)/lv-gpu.o: $(SRC_DIR)/lv-gpu.cu $(INC_DIR)/globals.h | $(OBJ_DIR)
	$(NVCC) -c -I$(INC_DIR) $(NVCCFLAGS) $< -o $@

std-params: $(TARGET_STDP)

$(TARGET_STDP): $(OBJ_DIR)/std-params.o | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/std-params.o: $(SRC_DIR)/std-params.cpp $(INC_DIR)/globals.h | $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@