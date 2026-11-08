.PHONY: all clean

TARGET 	 = build/lv
CXX 	 = g++
LD 	 = g++
CXXFLAGS = -O3 -march=native -Wall -Wextra -fopenmp
OBJS     = obj/lv.o

all: $(TARGET)

$(TARGET): $(OBJS)
	$(LD) $^ -o $@

obj/%.o: src/%.cpp
	$(CXX) -c $(CXXFLAGS) $< -o $@

clean:
	rm -f $(TARGET) obj/%.o
