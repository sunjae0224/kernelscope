// Reduced split-KV decode attention (fp32, GQA, d = 64) for GPGPU-Sim: reproduces how the number of KV
// splits spreads a ragged batch over thread blocks. Same structure as flash-attn's decode path:
// grid = (S, B, H_q); split s of every (b, h) covers keys [s*chunk, (s+1)*chunk) of the CACHE
// CAPACITY (max length), so splits past a short request's live length do no work; a second kernel
// merges the S partial (max, sum, acc) triples. Inside a block each warp walks keys; the 32 lanes
// share one key (2 dims each) so K/V rows are read as coalesced 256-byte lines.
// Usage: splitkv_attn B H_q H_kv S lens   (lens: "4096,128x15")
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cuda_runtime.h>

constexpr int D = 64, WARPS = 4, T = 32 * WARPS;

// Warp-wide sum through shared memory (GPGPU-Sim's shfl.sync model gave results off by ~1e-2 here).
__device__ inline float warp_sum(float v, float* red, int lane) {
    red[lane] = v;
    __syncwarp();
    for (int o = 16; o > 0; o >>= 1) {
        if (lane < o) red[lane] += red[lane + o];
        __syncwarp();
    }
    const float total = red[0];
    __syncwarp();
    return total;
}

__global__ void splitkv_partial(const float* q, const float* K, const float* V, const int* lens, int Lcap, int Hq, int Hkv,
                                int S, float* Opart, float* Mpart, float* Lpart, float* O) {
    const int s = blockIdx.x, b = blockIdx.y, h = blockIdx.z, hk = h / (Hq / Hkv);
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int chunk = (Lcap + S - 1) / S;
    const int lo = s * chunk, hi = min(lo + chunk, lens[b]);
    const float scale = rsqrtf((float)D);
    const float q0 = q[(b * Hq + h) * D + 2 * lane] * scale, q1 = q[(b * Hq + h) * D + 2 * lane + 1] * scale;
    __shared__ float red[WARPS][32];
    float m = -INFINITY, l = 0.f, a0 = 0.f, a1 = 0.f;
    for (int j = lo + warp; j < hi; j += WARPS) {                      // one key per warp iteration
        const size_t row = ((size_t)(b * Lcap + j) * Hkv + hk) * D + 2 * lane;
        const float2 k = *reinterpret_cast<const float2*>(K + row);
        const float2 v = *reinterpret_cast<const float2*>(V + row);
        const float sc = warp_sum(q0 * k.x + q1 * k.y, red[warp], lane);   // same value on every lane
        const float mn = fmaxf(m, sc), a = __expf(m - mn), p = __expf(sc - mn);
        l = l * a + p;
        a0 = a0 * a + p * v.x;
        a1 = a1 * a + p * v.y;
        m = mn;
    }
    __shared__ float sm[WARPS], sl[WARPS], sacc[WARPS][D];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
    sacc[warp][2 * lane] = a0; sacc[warp][2 * lane + 1] = a1;
    __syncthreads();
    if (warp == 0) {                                                    // merge the 4 warp states
        float mx = -INFINITY;
        for (int w = 0; w < WARPS; w++) mx = fmaxf(mx, sm[w]);
        float lsum = 0.f, o0 = 0.f, o1 = 0.f;
        for (int w = 0; w < WARPS; w++) {
            const float wgt = (sm[w] == -INFINITY) ? 0.f : __expf(sm[w] - mx);
            lsum += sl[w] * wgt;
            o0 += sacc[w][2 * lane] * wgt;
            o1 += sacc[w][2 * lane + 1] * wgt;
        }
        const size_t o = ((size_t)(s * gridDim.y + b) * Hq + h);
        if (S == 1) { O[(b * Hq + h) * D + 2 * lane] = o0 / lsum; O[(b * Hq + h) * D + 2 * lane + 1] = o1 / lsum; }
        else { Opart[o * D + 2 * lane] = o0; Opart[o * D + 2 * lane + 1] = o1; if (lane == 0) { Mpart[o] = mx; Lpart[o] = lsum; } }
    }
}

__global__ void splitkv_combine(const float* Opart, const float* Mpart, const float* Lpart, int S, int B, int Hq, float* O) {
    const int b = blockIdx.x, h = blockIdx.y, t = threadIdx.x;
    float mx = -INFINITY;
    for (int s = 0; s < S; s++) mx = fmaxf(mx, Mpart[((size_t)(s * B + b) * Hq + h)]);
    float l = 0.f, a = 0.f;
    for (int s = 0; s < S; s++) {
        const size_t o = ((size_t)(s * B + b) * Hq + h);
        const float w = (Mpart[o] == -INFINITY) ? 0.f : __expf(Mpart[o] - mx);
        l += Lpart[o] * w;
        a += Opart[o * D + t] * w;
    }
    O[(b * Hq + h) * D + t] = a / l;
}

static unsigned g_seed = 12345;
static float frand() { g_seed = g_seed * 1664525u + 1013904223u; return ((g_seed >> 8) & 0xFFFF) / 65536.0f - 0.5f; }

static std::vector<int> parse_lens(const char* s) {
    std::vector<int> out; std::string str(s); size_t p = 0;
    while (p < str.size()) {
        size_t c = str.find(',', p); std::string tok = str.substr(p, c == std::string::npos ? std::string::npos : c - p);
        size_t x = tok.find('x'); int len = atoi(tok.substr(0, x).c_str()), n = x == std::string::npos ? 1 : atoi(tok.substr(x + 1).c_str());
        for (int i = 0; i < n; i++) out.push_back(len);
        if (c == std::string::npos) break; p = c + 1;
    }
    return out;
}

int main(int argc, char** argv) {
    if (argc < 6) { fprintf(stderr, "usage: %s B H_q H_kv S lens\n", argv[0]); return 2; }
    const int B = atoi(argv[1]), Hq = atoi(argv[2]), Hkv = atoi(argv[3]), S = atoi(argv[4]);
    std::vector<int> lens = parse_lens(argv[5]);
    if ((int)lens.size() != B || Hq % Hkv) { fprintf(stderr, "bad shape\n"); return 2; }
    int Lcap = 0; for (int x : lens) Lcap = std::max(Lcap, x);
    const size_t nq = (size_t)B * Hq * D, nkv = (size_t)B * Lcap * Hkv * D;
    std::vector<float> q(nq), K(nkv), V(nkv), O(nq), ref(nq);
    for (auto& x : q) x = frand(); for (auto& x : K) x = frand(); for (auto& x : V) x = frand();
    for (int b = 0; b < B; b++) for (int h = 0; h < Hq; h++) {                  // CPU reference in double
        int hk = h / (Hq / Hkv); std::vector<double> sc(lens[b]); double mx = -1e300;
        for (int j = 0; j < lens[b]; j++) { double s = 0; for (int i = 0; i < D; i++) s += q[(b * Hq + h) * D + i] * K[((size_t)(b * Lcap + j) * Hkv + hk) * D + i]; sc[j] = s / sqrt((double)D); mx = std::max(mx, sc[j]); }
        double l = 0; std::vector<double> acc(D, 0.0);
        for (int j = 0; j < lens[b]; j++) { double p = exp(sc[j] - mx); l += p; for (int i = 0; i < D; i++) acc[i] += p * V[((size_t)(b * Lcap + j) * Hkv + hk) * D + i]; }
        for (int i = 0; i < D; i++) ref[(b * Hq + h) * D + i] = (float)(acc[i] / l);
    }
    float *dq, *dK, *dV, *dO, *dOp, *dM, *dL; int* dlens;
    cudaMalloc(&dq, nq * 4); cudaMalloc(&dK, nkv * 4); cudaMalloc(&dV, nkv * 4); cudaMalloc(&dO, nq * 4);
    cudaMalloc(&dOp, (size_t)S * nq * 4); cudaMalloc(&dM, (size_t)S * B * Hq * 4); cudaMalloc(&dL, (size_t)S * B * Hq * 4); cudaMalloc(&dlens, B * 4);
    cudaMemcpy(dq, q.data(), nq * 4, cudaMemcpyHostToDevice); cudaMemcpy(dK, K.data(), nkv * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, V.data(), nkv * 4, cudaMemcpyHostToDevice); cudaMemcpy(dlens, lens.data(), B * 4, cudaMemcpyHostToDevice);
    splitkv_partial<<<dim3(S, B, Hq), T>>>(dq, dK, dV, dlens, Lcap, Hq, Hkv, S, dOp, dM, dL, dO);
    if (S > 1) splitkv_combine<<<dim3(B, Hq), D>>>(dOp, dM, dL, S, B, Hq, dO);
    cudaMemcpy(O.data(), dO, nq * 4, cudaMemcpyDeviceToHost);
    double err = 0; for (size_t i = 0; i < nq; i++) err = std::max(err, (double)fabs(O[i] - ref[i]));
    printf("check B=%d Hq=%d Hkv=%d S=%d Lcap=%d ctas=%d max_abs_err=%.3e %s\n", B, Hq, Hkv, S, Lcap, S * B * Hq, err, err < 1e-3 ? "ok" : "FAIL");
    return err < 1e-3 ? 0 : 1;
}
