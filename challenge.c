#include <stdio.h>
#include <stdlib.h>

/**
 * Perform Sparse Matrix-Vector Multiplication using CSR format.
 * 
 * Scans dense matrix A, builds CSR representation using caller-provided
 * buffers, and computes the matrix-vector product y = A * x.
 *
 * CRITICAL CONSTRAINT: ZERO dynamic memory allocation.
 *
 * @param rows    Number of rows in matrix A
 * @param cols    Number of columns in matrix A
 * @param A       Dense matrix A (1D array of size rows*cols, row-major)
 * @param x       Input vector x (size cols)
 * @param values  Pre-allocated buffer for CSR non-zero values
 * @param col_idx Pre-allocated buffer for CSR column indices
 * @param row_ptr Pre-allocated buffer for CSR row pointers
 * @param y       Pre-allocated buffer for the output vector
 */
void sparse_multiply(int rows, int cols, float *A, float *x, float *values, int *col_idx, int *row_ptr, float *y) {
    int nnz = 0;

    // Phase 1: Build CSR Representation
    for (int i = 0; i < rows; i++) {
        // Store starting index of this row's data
        row_ptr[i] = nnz;
        
        // Optimization 1: Store pointer to the row to avoid redundant multiplications (i * cols)
        float *row = &A[i * cols]; 
        
        // Optimization 2: Sequential memory access for cache-friendly traversal
        for (int j = 0; j < cols; j++) {
            if (row[j] != 0.0f) {
                values[nnz] = row[j];
                col_idx[nnz] = j;
                nnz++;
            }
        }
    }
    
    // Finalize row_ptr
    row_ptr[rows] = nnz;

    // Phase 2: Sparse Matrix Vector Multiply
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        
        // Loop over non-zero elements in row i
        for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
            sum += values[k] * x[col_idx[k]];
        }
        
        y[i] = sum;
    }
}

// Test Harness
int main() {
    // 4x4 dense matrix A
    int rows = 4;
    int cols = 4;
    float A[] = {
        1.0f, 0.0f, 4.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 3.0f, 5.0f, 0.0f,
        2.0f, 0.0f, 6.0f, 0.0f
    };
    
    // Input vector x
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Pre-allocate arrays to avoid dynamic allocation.
    // In a worst-case scenario (dense), nnz could be up to rows * cols = 16.
    float values[16];
    int col_idx[16];
    int row_ptr[5]; // rows + 1
    float y[4];
    
    printf("Starting sparse matrix-vector multiplication...\n");
    sparse_multiply(rows, cols, A, x, values, col_idx, row_ptr, y);
    
    printf("\nResult y = A * x:\n");
    for (int i = 0; i < rows; i++) {
        printf("y[%d] = %.2f\n", i, y[i]);
    }
    
    printf("\nExpected y:\n");
    printf("y[0] = 13.00\n");
    printf("y[1] = 0.00\n");
    printf("y[2] = 21.00\n");
    printf("y[3] = 20.00\n");

    return 0;
}
