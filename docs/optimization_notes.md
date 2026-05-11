# Optimization Notes

This kernel incorporates several performance-conscious optimizations specifically chosen for efficient execution on low-level system hardware.

## 1. Cache Locality and Sequential Memory Access
The CPU cache is most efficient when it loads data sequentially. Sparse representations like Linked Lists cause pointer-chasing and cache misses. 
The CSR layout stores `values[]` and `col_idx[]` as 1D contiguous arrays. When our `sparse_matvec()` function loops over a row, it traverses these arrays sequentially. This allows the CPU hardware prefetcher to load upcoming values into L1/L2 cache before they are needed, resulting in extremely high cache hit rates.

## 2. Sparse Traversal Efficiency
By maintaining `row_ptr[]`, the kernel completely skips rows that contain all zeros without executing any loops or conditionals. When multiplying, the inner loop bounds (`row_start` to `row_end`) tightly constrain operations to non-zero values only. 

## 3. Branch Reduction
A naive sparse matrix implementation might iterate over all elements and use `if (val != 0)` to decide whether to multiply. Branch instructions are expensive and can disrupt the CPU pipeline. The CSR layout inherently encodes the structure of the data, eliminating the need for runtime conditionals in the multiplication loop.

## 4. Why CSR is Efficient
CSR strikes the perfect balance between memory compression and fast arithmetic. It compresses the row overhead into a minimal index array (`row_ptr`) while allowing the matrix-vector dot product to map perfectly to tight, predictable loops.
