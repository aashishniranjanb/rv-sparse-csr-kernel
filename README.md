# Sparse Matrix Vector Multiplication using CSR Representation in C

## Project Overview
This project implements sparse matrix-vector multiplication using the Compressed Sparse Row (CSR) representation in C without any dynamic memory allocation. The goal is to efficiently scan a dense row-major matrix, extract its non-zero elements into the CSR format using pre-allocated buffers, and compute the matrix-vector product $y = A \times x$. 

This project adheres to strict systems programming discipline and emphasizes memory efficiency.

## Features
- **Dense to CSR conversion**: Efficient row-by-row scanning to extract non-zero values.
- **Sparse matrix-vector multiplication**: Calculates results using only non-zero elements.
- **Zero dynamic memory allocation**: Strict adherence to pre-allocated buffers on the stack.
- **Cache-friendly traversal**: Uses pointers to row starts and iterates sequentially to maximize cache hit rates.
- **Clean modular implementation**: Highly readable, modular C code written with future extensibility in mind.

## CSR Explanation
The Compressed Sparse Row (CSR) format represents a sparse matrix efficiently by storing only its non-zero elements.

A matrix:
```
1 0 4 0
0 0 0 0
0 3 5 0
2 0 6 0
```
Is represented by three arrays in CSR:
- `values[]`: `[1, 4, 3, 5, 2, 6]` (Stores non-zero values)
- `col_idx[]`: `[0, 2, 1, 2, 0, 2]` (Stores column indices)
- `row_ptr[]`: `[0, 2, 2, 4, 6]` (Where each row starts in `values[]`)

For a deeper dive, check out the [CSR Explanation Documentation](docs/csr_explanation.md).

## Build Instructions
A standard Makefile is provided for compilation.
```bash
make
./run
```
Or manually:
```bash
gcc -O2 -lm -o run challenge.c
./run
```

## Complexity Analysis
- **Dense matrix multiplication**: $O(\text{rows} \times \text{cols})$
- **CSR matrix multiplication**: $O(nnz)$, where $nnz$ is the total number of non-zero elements.

This dramatically reduces redundant computations on matrices with high sparsity. For performance analysis, see [Benchmark Notes](benchmark/notes.md).

## Future Work
There are several ways to expand upon this systems-level implementation, particularly in the realm of high-performance computing (HPC):
- **RISC-V Vector intrinsics (RVV)**: Exploring vectorized dot products using RVV for native hardware acceleration.
- **SIMD (Single Instruction Multiple Data)**: Implementing standard SIMD acceleration for the matrix multiplications.
- **Cache-aware traversal**: Further tuning loop unrolling or software prefetching.
- **Sparse Neural Networks**: Adapting this kernel to efficiently execute sparse AI workloads.
