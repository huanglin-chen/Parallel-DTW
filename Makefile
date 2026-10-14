CXX = g++
NVCC = nvcc
CXXFLAGS = -Wall
NVCCFLAGS = -rdc=true

TARGET = src/main
OBJS = src/main.o src/simple.o src/diagonal.o src/prefix.o src/hybrid.o

all: $(TARGET)

run:
	$(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

src/main.o: src/main.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/simple.o: src/simple.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

src/diagonal.o: src/diagonal.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

src/prefix.o: src/prefix.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

src/hybrid.o: src/hybrid.cu
	$(NVCC) $(NVCCFLAGS) -c -o $@ $<

.PHONY: clean

clean:
	rm -f $(OBJS) $(TARGET)