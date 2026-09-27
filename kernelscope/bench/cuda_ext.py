"""Small CUDA kernels Triton cannot express: an SM blocker and an %smid probe.

Built once with torch.utils.cpp_extension.load_inline and the CUDA 12.9 toolchain of the
accelsim-build conda env: /usr/bin/nvcc is CUDA 10.1 and cannot target sm_89.
"""
import os
from pathlib import Path

CUDA_HOME = os.environ.get("KERNELSCOPE_CUDA_HOME", "/home/skkai/miniforge3/envs/accelsim-build")

CPP = """
void launch_blocker(int64_t n_ctas, int64_t smem_bytes, int64_t spin_cycles, int64_t stream_ptr);
torch::Tensor smid_probe(int64_t n_ctas, int64_t threads, int64_t smem_bytes, int64_t spin_cycles);
"""

CUDA = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void __launch_bounds__(32) ks_sm_blocker(long long spin_cycles) {
    extern __shared__ char smem[];
    if (threadIdx.x == 0) smem[0] = 0;
    __syncthreads();
    long long start = clock64();
    while (clock64() - start < spin_cycles) { }
}

__global__ void ks_smid_probe(int* out, long long spin_cycles) {
    extern __shared__ char smem[];
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        out[blockIdx.x] = (int)smid;
        smem[0] = 0;
    }
    __syncthreads();
    long long start = clock64();
    while (clock64() - start < spin_cycles) { }
}

void launch_blocker(int64_t n_ctas, int64_t smem_bytes, int64_t spin_cycles, int64_t stream_ptr) {
    if (n_ctas <= 0) return;
    auto stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    cudaFuncSetAttribute(ks_sm_blocker, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    ks_sm_blocker<<<(unsigned)n_ctas, 32, (size_t)smem_bytes, stream>>>((long long)spin_cycles);
}

torch::Tensor smid_probe(int64_t n_ctas, int64_t threads, int64_t smem_bytes, int64_t spin_cycles) {
    auto out = torch::full({n_ctas}, -1, torch::dtype(torch::kInt32).device(torch::kCUDA));
    cudaFuncSetAttribute(ks_smid_probe, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    auto stream = c10::cuda::getCurrentCUDAStream();
    ks_smid_probe<<<(unsigned)n_ctas, (unsigned)threads, (size_t)smem_bytes, stream>>>(
        out.data_ptr<int>(), (long long)spin_cycles);
    return out;
}
"""

_EXT = None


def load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT
    import torch
    import torch.utils.cpp_extension as ce
    major, minor = torch.cuda.get_device_capability()
    os.environ["CUDA_HOME"] = CUDA_HOME
    os.environ["PATH"] = f"{CUDA_HOME}/bin:" + os.environ["PATH"]
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    ce.CUDA_HOME = CUDA_HOME                      # cpp_extension caches CUDA_HOME at import time
    build = Path.home() / ".cache" / "kernelscope" / f"cuda_ext_sm{major}{minor}"
    build.mkdir(parents=True, exist_ok=True)
    _EXT = ce.load_inline(
        name="kernelscope_cuda_ext", cpp_sources=CPP, cuda_sources=CUDA,
        functions=["launch_blocker", "smid_probe"],
        extra_cuda_cflags=["-ccbin", f"{CUDA_HOME}/bin/x86_64-conda-linux-gnu-g++", f"-arch=sm_{major}{minor}"],
        build_directory=str(build), verbose=False,
    )
    return _EXT
