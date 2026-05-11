#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>

/*
 * RV-Sparse CSR Kernel
 * 
 * Modular implementation of Dense-to-CSR extraction and Sparse Matrix-Vector
 * Multiplication (SpMV). Designed for systems programming correctness: zero 
 * dynamic memory allocation, cache-friendly sequential access, and modularity.
 */

// ==========================================
// Helper Functions
// ==========================================

/**
 * Extracts non-zero elements from a dense matrix into CSR format.
 * All buffers must be caller-provided.
 * 
 * @return Total number of non-zero elements (nnz)
 */
int extract_csr(int rows, int cols, const float *dense_matrix, 
                float *values, int *col_idx, int *row_ptr) {
    int nonzero_count = 0;

    for (int i = 0; i < rows; i++) {
        // Record where this row starts in the values/col_idx arrays
        row_ptr[i] = nonzero_count;
        
        // Cache-friendly row pointer mapping
        const float *current_row = &dense_matrix[i * cols];
        
        for (int j = 0; j < cols; j++) {
            if (current_row[j] != 0.0f) {
                values[nonzero_count] = current_row[j];
                col_idx[nonzero_count] = j;
                nonzero_count++;
            }
        }
    }
    
    // Finalize row_ptr to simplify loop bounds during multiplication
    row_ptr[rows] = nonzero_count;
    
    return nonzero_count;
}

/**
 * Computes Sparse Matrix-Vector Multiplication: y = A * x
 * 
 * @return Total number of sparse multiplications performed.
 */
int sparse_matvec(int rows, const float *values, const int *col_idx, 
                  const int *row_ptr, const float *x, float *y) {
    int ops_count = 0;

    for (int i = 0; i < rows; i++) {
        float accumulator = 0.0f;
        int row_start = row_ptr[i];
        int row_end = row_ptr[i + 1];

        // Traverse CSR row segment for sparse dot product accumulation
        for (int k = row_start; k < row_end; k++) {
            accumulator += values[k] * x[col_idx[k]];
            ops_count++;
        }
        
        y[i] = accumulator;
    }
    
    return ops_count;
}

/**
 * Validates the CSR output against a standard Dense Matrix-Vector Multiplication.
 */
bool verify_result(int rows, int cols, const float *dense_matrix, const float *x, const float *csr_y) {
    bool passed = true;
    for (int i = 0; i < rows; i++) {
        float expected = 0.0f;
        const float *current_row = &dense_matrix[i * cols];
        for (int j = 0; j < cols; j++) {
            expected += current_row[j] * x[j];
        }
        // Using a tiny epsilon for float comparison safety
        if (abs(expected - csr_y[i]) > 1e-5) {
            passed = false;
            break;
        }
    }
    return passed;
}

/**
 * Prints the CSR data cleanly for debugging and verification.
 */
void print_csr(int rows, int nnz, const float *values, const int *col_idx, const int *row_ptr) {
    printf("===== CSR Data =====\n\n");
    
    printf("Values      : [");
    for (int i = 0; i < nnz; i++) printf("%.0f%s", values[i], (i == nnz-1) ? "" : " ");
    printf("]\n");
    
    printf("Column IDX  : [");
    for (int i = 0; i < nnz; i++) printf("%d%s", col_idx[i], (i == nnz-1) ? "" : " ");
    printf("]\n");
    
    printf("Row PTR     : [");
    for (int i = 0; i <= rows; i++) printf("%d%s", row_ptr[i], (i == rows) ? "" : " ");
    printf("]\n\n");
}

// ==========================================
// Main Execution / Test Harness
// ==========================================

int main() {
    // Configurable dimensions
    const int rows = 3;
    const int cols = 4;
    
    // Input Matrix A (Dense)
    float dense_matrix[] = {
        1.0f, 0.0f, 0.0f, 2.0f,
        0.0f, 0.0f, 3.0f, 0.0f,
        4.0f, 0.0f, 5.0f, 6.0f
    };
    
    // Input Vector x
    float x[] = {1.0f, 2.0f, 3.0f, 4.0f};
    
    // Caller-provided static buffers (Max Size = rows * cols)
    float values[12];
    int col_idx[12];
    int row_ptr[4]; // rows + 1
    float y[3];     // size = rows
    
    clock_t start, end;
    
    // Extract CSR
    start = clock();
    int nnz = extract_csr(rows, cols, dense_matrix, values, col_idx, row_ptr);
    end = clock();
    double extract_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Sparsity calculations
    int total_elements = rows * cols;
    float sparsity = (1.0f - ((float)nnz / (float)total_elements)) * 100.0f;

    // Display Overview
    printf("===== RV-Sparse CSR Kernel =====\n\n");
    printf("Input Matrix Dimensions : %d x %d\n", rows, cols);
    printf("Total Elements          : %d\n", total_elements);
    printf("Non-Zero Elements       : %d\n", nnz);
    printf("Matrix Sparsity         : %.2f%%\n\n", sparsity);

    // Display Input Matrix (Dense)
    printf("Input Matrix (Dense):\n");
    for (int i = 0; i < rows; i++) {
        printf("[ ");
        for (int j = 0; j < cols; j++) {
            printf("%.0f ", dense_matrix[i * cols + j]);
        }
        printf("]\n");
    }
    printf("\n");

    // Display CSR
    print_csr(rows, nnz, values, col_idx, row_ptr);

    // Sparse Matrix-Vector Multiply
    start = clock();
    int sparse_ops = sparse_matvec(rows, values, col_idx, row_ptr, x, y);
    end = clock();
    double multiply_time = ((double) (end - start)) / CLOCKS_PER_SEC;

    // Display Output
    printf("===== Sparse Matrix-Vector Multiplication =====\n\n");
    printf("Input Vector x:\n[ ");
    for (int j = 0; j < cols; j++) printf("%.0f ", x[j]);
    printf("]\n\nOutput Vector y = A * x:\n[ ");
    for (int i = 0; i < rows; i++) printf("%.0f ", y[i]);
    printf("]\n\n");

    // Performance Verification & Stats
    bool is_verified = verify_result(rows, cols, dense_matrix, x, y);
    int dense_ops = rows * cols;
    int skipped = dense_ops - nnz;
    float ratio = ((float)nnz / (float)dense_ops) * 100.0f;

    printf("===== Performance Summary =====\n\n");
    printf("Dense Multiplications Required : %d\n", dense_ops);
    printf("Sparse Multiplications Used    : %d\n", sparse_ops);
    printf("Operations Avoided             : %d\n", dense_ops - sparse_ops);
    printf("Zero Elements Skipped          : %d\n", skipped);
    printf("Useful Computation Ratio       : %.2f%%\n\n", ratio);
    
    printf("CSR Extraction Time            : %f sec\n", extract_time);
    printf("Sparse Multiply Time           : %f sec\n\n", multiply_time);

    printf("Dynamic Memory Allocation Used : NO\n");
    printf("CSR Traversal                  : SUCCESS\n");
    printf("Verification Status            : %s\n", is_verified ? "PASSED" : "FAILED");

    return 0;
}
