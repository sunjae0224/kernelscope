#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
  std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  constexpr int n = 1 << 20;
  cudaDeviceProp prop{};
  CUDA(cudaGetDeviceProperties(&prop, 0));
  std::printf("GPU=%s CC=%d.%d SMs=%d L2=%d regs=%d shared=%zu optin=%zu threads=%d\n",
      prop.name, prop.major, prop.minor, prop.multiProcessorCount, prop.l2CacheSize,
      prop.regsPerMultiprocessor, prop.sharedMemPerMultiprocessor,
      prop.sharedMemPerBlockOptin, prop.maxThreadsPerMultiProcessor);
  std::vector<float> a(n, 1), b(n, 2), c(n);
  float *da, *db, *dc;
  CUDA(cudaMalloc(&da, n * sizeof(float)));
  CUDA(cudaMalloc(&db, n * sizeof(float)));
  CUDA(cudaMalloc(&dc, n * sizeof(float)));
  CUDA(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CUDA(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  vecAdd<<<n / 256, 256>>>(da, db, dc, n);
  CUDA(cudaGetLastError());
  CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
  for (float v : c) if (v != 3) return 2;
  CUDA(cudaFree(da)); CUDA(cudaFree(db)); CUDA(cudaFree(dc));
  std::printf("PASS: vecadd result correct (N=%d)\n", n);
}
