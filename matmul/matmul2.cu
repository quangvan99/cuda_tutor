#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "../ops.h"
#define M 256
#define K 512
#define N 256   
#define BLOCK_SIZE 16


// CPU matrix multiplication
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// CUDA kernel for matrix multiplication
__global__ void _matmul_gpu(float *A, float *B, float *C, int m, int k, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int l = 0; l < k; l++) {
            sum += A[row * k + l] * B[l * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Initialize matrix with random values
void init_matrix(float *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = (float)rand() / RAND_MAX;
    }
}

void matmul_gpu(float *d_A, float *d_B, float *d_C, float *h_A, float *h_B, float *h_C, int m, int k, int n) {
    int size_A = m * k * sizeof(float);
    int size_B = k * n * sizeof(float);
    int size_C = m * n * sizeof(float);
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    // Define grid and block dimensions
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks(ceil(n / BLOCK_SIZE), ceil(m / BLOCK_SIZE));
    _matmul_gpu<<<blocks, threads>>>(d_A, d_B, d_C, m, k, n);
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
}

int main() {
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;
    int size_A = M * K * sizeof(float);
    int size_B = K * N * sizeof(float);
    int size_C = M * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    // Initialize matrices
    srand(time(NULL));
    init_matrix(h_A, M, K);
    init_matrix(h_B, K, N);

    // Allocate device memory
    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);


    // Warm-up runs
    printf("Performing warm-up runs...\n");
    for (int i = 0; i < 3; i++) {
        matmul_cpu(h_A, h_B, h_C, M, K, N);
        matmul_gpu(d_A, d_B, d_C, h_A, h_B, h_C, M, K, N);
        cudaDeviceSynchronize();
    }

    printf("Benchmarking CPU implementation...\n");
    measure_exec_time(matmul_cpu, h_A, h_B, h_C, M, K, N);

    printf("Benchmarking GPU implementation...\n");
    measure_exec_time(matmul_gpu, d_A, d_B, d_C, h_A, h_B, h_C, M, K, N);

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}