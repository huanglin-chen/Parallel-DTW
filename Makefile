CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall -Wextra -O2
NVCCFLAGS = -rdc=true -O2

TARGET = src/main
OBJS = src/main.o src/simple.o src/diagonal.o src/prefix.o src/hybrid.o

.PHONY: all run clean

all: $(TARGET)

run: $(TARGET)
	$(TARGET)

clean:
	rm -f $(OBJS) $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/%.o: src/%.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<