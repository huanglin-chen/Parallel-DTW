# Parallel-DTW

## Description

This project explores GPU acceleration of Dynamic Time Warping (DTW) using NVIDIA CUDA. It implements and compares three parallelization strategies for computing the DTW cost matrix:

- Diagonal (Wavefront): Parallelizes the computation along anti-diagonals of the cost matrix.

- Prefix Sum: Uses prefix computations to transform data dependencies and enable row-wise parallelization.

- Hybrid: Combines diagonal computation with prefix-based processing.

The project includes sequential and CUDA implementations, correctness verification, and performance benchmarking using time series datasets from the UCR Time Series Classification Archive.

## Requirements

- OS: Ubuntu 24.04 (Tested Environment)
- GPU: NVIDIA GPU with CUDA support
- CUDA Toolkit: 13.0 (Tested Environment)
- C++ Compiler: GCC/G++ compatible with the CUDA Toolkit
- Build Tools: GNU Make

## Installation

### Prerequisites

1. Install the build tools:
```
sudo apt update
sudo apt install build-essential
```
2. Install the NVIDIA CUDA Toolkit following the official installation guide and verify:
```
nvidia-smi
nvcc --version
```

### Clone the Repository

```
git clone https://github.com/huanglin-chen/Parallel-DTW.git
cd Parallel-DTW
```

### Build

```
make
```

## Usage

After building the project, run the program using:
```
make run
```
This command executes the main program (src/main), which runs the DTW experiments.

To remove generated build files, run:
```
make clean
```

## Benchmarks

The project includes the following benchmarks:

- **Cost Matrix**: End-to-end execution time of computing the cost matrix using CPU sequential and GPU parallel implementations.

- **Kernel**: execution time of CUDA kernels, excluding memory allocation and data transfers.

- **Stream**: End-to-end execution time of computing the cost matrix per sequence pair using CUDA Streams.


## References

- Xiao, L., Zheng, Y., Tang, W., Yao, G., & Ruan, L. (2013). Parallelizing Dynamic Time Warping Algorithm Using Prefix Computations on GPU. IEEE HPCC/EUC 2013. https://doi.org/10.1109/HPCC.and.EUC.2013.50
- [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data/)