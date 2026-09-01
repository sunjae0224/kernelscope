// Naive decode attention as a standalone CUDA binary — an example "someone else's kernel"
// driven through kernelscope's ExecutablePlugin contract:
//
//   naive_attn <B> <L_kv> <H_q> <H_kv> <d> <iters> [out_path]
//
//   * runs the kernel <iters> times, timing each with CUDA events, and prints one line
//       KERNELSCOPE {"kernel_time_us": <median>, "launches_per_iter": 1}
//   * if out_path is given, writes the output [B, 1, H_q, d] as raw float32.
//
// Inputs are generated deterministically from a hash of the flat index (see init()),
// which the plugin reproduces in torch for the correctness check.
//
// One thread block per (query head, batch); two passes over the keys (max, then
// exp-sum + weighted V) — no tensor cores, no tiling. Deliberately simple.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);} } while (0)

// deterministic pseudo-random init in [-0.5, 0.5); identical in the plugin (torch, int64)
__host__ __device__ inline float init_val(unsigned long long i, unsigned int seed) {
    unsigned int x = (unsigned int)(i + (unsigned long long)seed * 1000003ull) * 2654435761u;
    return (float)(x & 0xffffu) / 65536.0f - 0.5f;
}

__global__ void init_kernel(float* p, unsigned long long n, unsigned int seed) {
    unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) p[i] = init_val(i, seed);
}

// q: [B, H_q, d]   k, v: [B, L_kv, H_kv, d]   o: [B, H_q, d]
__global__ void naive_decode_attn(const float* __restrict__ q, const float* __restrict__ k,
                                  const float* __restrict__ v, float* __restrict__ o,
                                  int L_kv, int H_q, int H_kv, int d) {
    const int h = blockIdx.x, b = blockIdx.y;
    const int hk = h / (H_q / H_kv);
    const float scale = rsqrtf((float)d);
    const float* qv = q + ((size_t)b * H_q + h) * d;
    const float* kb = k + (size_t)b * L_kv * H_kv * d;
    const float* vb = v + (size_t)b * L_kv * H_kv * d;

    __shared__ float red[128];
    __shared__ float acc_s[256];
    for (int i = threadIdx.x; i < d; i += blockDim.x) acc_s[i] = 0.f;

    // pass 1: max score
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L_kv; j += blockDim.x) {
        const float* kj = kb + ((size_t)j * H_kv + hk) * d;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += qv[t] * kj[t];
        m = fmaxf(m, s * scale);
    }
    red[threadIdx.x] = m;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
        __syncthreads();
    }
    m = red[0];
    __syncthreads();

    // pass 2: exp-sum and weighted V
    float l = 0.f;
    float acc[256];
    for (int t = 0; t < d; ++t) acc[t] = 0.f;
    for (int j = threadIdx.x; j < L_kv; j += blockDim.x) {
        const float* kj = kb + ((size_t)j * H_kv + hk) * d;
        const float* vj = vb + ((size_t)j * H_kv + hk) * d;
        float s = 0.f;
        for (int t = 0; t < d; ++t) s += qv[t] * kj[t];
        float p = __expf(s * scale - m);
        l += p;
        for (int t = 0; t < d; ++t) acc[t] += p * vj[t];
    }
    red[threadIdx.x] = l;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
        __syncthreads();
    }
    l = red[0];
    for (int t = 0; t < d; ++t) atomicAdd(&acc_s[t], acc[t]);
    __syncthreads();
    float* ov = o + ((size_t)b * H_q + h) * d;
    for (int t = threadIdx.x; t < d; t += blockDim.x) ov[t] = acc_s[t] / l;
}

int main(int argc, char** argv) {
    if (argc < 7) { fprintf(stderr, "usage: %s B L_kv H_q H_kv d iters [out_path]\n", argv[0]); return 2; }
    const int B = atoi(argv[1]), L = atoi(argv[2]), Hq = atoi(argv[3]), Hkv = atoi(argv[4]), d = atoi(argv[5]);
    const int iters = atoi(argv[6]);
    const char* out_path = argc > 7 ? argv[7] : nullptr;
    if (d > 256 || Hq % Hkv) { fprintf(stderr, "d <= 256 and H_q %% H_kv == 0 required\n"); return 2; }

    const unsigned long long nq = (unsigned long long)B * Hq * d, nkv = (unsigned long long)B * L * Hkv * d;
    float *q, *k, *v, *o;
    CHECK(cudaMalloc(&q, nq * sizeof(float)));
    CHECK(cudaMalloc(&k, nkv * sizeof(float)));
    CHECK(cudaMalloc(&v, nkv * sizeof(float)));
    CHECK(cudaMalloc(&o, nq * sizeof(float)));
    init_kernel<<<(unsigned)((nq + 255) / 256), 256>>>(q, nq, 1);
    init_kernel<<<(unsigned)((nkv + 255) / 256), 256>>>(k, nkv, 2);
    init_kernel<<<(unsigned)((nkv + 255) / 256), 256>>>(v, nkv, 3);
    CHECK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));
    std::vector<float> times;
    dim3 grid(Hq, B), block(128);
    for (int it = 0; it < iters; ++it) {
        CHECK(cudaEventRecord(e0));
        naive_decode_attn<<<grid, block>>>(q, k, v, o, L, Hq, Hkv, d);
        CHECK(cudaEventRecord(e1));
        CHECK(cudaEventSynchronize(e1));
        float ms; CHECK(cudaEventElapsedTime(&ms, e0, e1));
        times.push_back(ms * 1000.f);
    }
    CHECK(cudaGetLastError());
    std::sort(times.begin(), times.end());
    const float med = times[times.size() / 2];

    if (out_path) {
        std::vector<float> host(nq);
        CHECK(cudaMemcpy(host.data(), o, nq * sizeof(float), cudaMemcpyDeviceToHost));
        FILE* f = fopen(out_path, "wb");
        if (!f) { fprintf(stderr, "cannot open %s\n", out_path); return 1; }
        fwrite(host.data(), sizeof(float), nq, f);
        fclose(f);
    }
    printf("KERNELSCOPE {\"kernel_time_us\": %.3f, \"launches_per_iter\": 1, \"iters\": %d}\n", med, iters);
    return 0;
}
