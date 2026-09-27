#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
__global__ void vecadd(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
int main() {
    const int n = 1 << 16;
    float *a = (float*)malloc(n * 4), *b = (float*)malloc(n * 4), *c = (float*)malloc(n * 4);
    for (int i = 0; i < n; i++) { a[i] = i * 0.5f; b[i] = 1.0f; }
    float *da, *db, *dc;
    cudaMalloc(&da, n * 4); cudaMalloc(&db, n * 4); cudaMalloc(&dc, n * 4);
    cudaMemcpy(da, a, n * 4, cudaMemcpyHostToDevice); cudaMemcpy(db, b, n * 4, cudaMemcpyHostToDevice);
    vecadd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    cudaMemcpy(c, dc, n * 4, cudaMemcpyDeviceToHost);
    int bad = 0; for (int i = 0; i < n; i++) if (c[i] != a[i] + b[i]) bad++;
    printf("vecadd n=%d mismatches=%d\n", n, bad);
    return bad ? 1 : 0;
}
