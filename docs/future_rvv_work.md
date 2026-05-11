# Future RVV Work

As this project evolves toward high-performance computing, the next logical step is porting the scalar C loops to utilize **RISC-V Vector (RVV)** intrinsics.

## 1. RVV Vector Intrinsics
RISC-V Vector extension allows single instructions to operate on multiple data elements simultaneously. The sparse multiplication inner loop can be transformed to load a vector of `values` and perform vectorized fused-multiply-adds (FMA).

## 2. Gather Loads (`vluxei`)
The primary challenge of vectorized sparse computations is non-contiguous memory access on the vector `x`. 
While `values[]` is accessed sequentially, the input vector `x[col_idx[k]]` is accessed randomly. RVV provides "gather loads" (e.g., `vluxei32.v`) which allow the hardware to fetch disjoint memory locations simultaneously into a vector register based on the indices stored in `col_idx[]`.

## 3. Vectorized Sparse Kernels & SIMD Acceleration
By combining gather loads with vector FMAs, we can accelerate the `sparse_matvec` routine. If multiple rows have similar lengths, they can also be batched using segment loads.

## 4. Sparse AI Inference
Pruned neural networks naturally form sparse matrices. Integrating an RVV-accelerated CSR kernel into edge AI frameworks (like TFLite Micro) allows RISC-V IoT chips to execute massive machine learning models with minimal power and memory.
