.PHONY: all clean

CXX 	 = g++
LD 	     = g++
CXXFLAGS = -O3 -march=native -Wall -Wextra -fopenmp
LDFLAGS  = -fopenmp
OBJ_DIR  = obj
BIN_DIR  = build
OBJS     = $(OBJ_DIR)/lv.o
TARGET 	 = $(BIN_DIR)/lv

all: $(TARGET)

$(TARGET): $(OBJS) | $(BIN_DIR)
	$(LD) $(LDFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: src/%.cpp | $(OBJ_DIR)
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

$(OBJ_DIR) $(BIN_DIR):
	mkdir -p $@