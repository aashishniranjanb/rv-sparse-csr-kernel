# Benchmark and Complexity Notes

This document provides a theoretical benchmark comparison between Dense Matrix-Vector Multiplication and Sparse Matrix-Vector Multiplication using the CSR format.

## Time Complexity

### 1. Dense Matrix Multiplication
When computing $y = A \times x$ on a dense matrix, we must perform an arithmetic multiplication and addition for every single cell in the matrix, regardless of whether the cell contains a zero.
- **Operations:** For each row, iterate through all columns.
- **Complexity:** $O(\text{rows} \times \text{cols})$
- **Instruction Overhead:** Heavy branching is not typically present, but the pure volume of floating-point operations makes it slow.

### 2. CSR Matrix Multiplication
When using CSR, the algorithm strictly iterates over the `nnz` (number of non-zero elements). The zeros are entirely ignored.
- **Operations:** For each row, iterate from `row_ptr[i]` to `row_ptr[i+1]`.
- **Complexity:** $O(nnz)$

### Theoretical Example
Consider a $10,000 \times 10,000$ matrix with a sparsity of 1% (meaning 99% of the values are zero).
- **Dense Operations:** $10,000 \times 10,000 = 100,000,000$ calculations.
- **CSR Operations:** $1\% \times 100,000,000 = 1,000,000$ calculations.

The CSR approach requires **100x fewer operations**, dramatically speeding up inference workloads and conserving energy on embedded systems.

## Memory Complexity

- **Dense Matrix:** $O(\text{rows} \times \text{cols})$ space.
- **CSR:** Space is bounded by $2 \times nnz + \text{rows} + 1$. For sparse matrices, this represents an enormous reduction in memory overhead.
