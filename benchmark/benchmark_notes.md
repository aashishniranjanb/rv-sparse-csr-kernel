# Benchmark Notes

Understanding the exact benefit of sparse operations requires comparing theoretical operation counts across different matrix dimensions and sparsity levels.

## Performance Analysis

| Matrix Size | Total Elements | Sparsity | Dense Ops (Multiplications) | Sparse Ops (CSR) | Operations Avoided |
|-------------|----------------|----------|-----------------------------|------------------|--------------------|
| 3x4         | 12             | 50%      | 12                          | 6                | 6                  |
| 100x100     | 10,000         | 88%      | 10,000                      | 1,200            | 8,800              |
| 500x500     | 250,000        | 94%      | 250,000                     | 14,000           | 236,000            |

### Analysis
As the matrix size increases, AI and scientific matrices tend to become extremely sparse (often $>90\%$ sparsity in pruned neural networks).
- **Dense Ops** scale linearly with `rows * cols`, unconditionally calculating `0.0 * x`.
- **Sparse Ops** completely skip zero values, directly mirroring the $O(nnz)$ complexity.

By utilizing the CSR layout, the computation dynamically scales down to the actual "useful computation ratio," preventing massive battery drain and thermal throttling on embedded systems.
