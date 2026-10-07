#pragma once
// Programmatic dependent launch (sm_90+): a weight GEMM whose WEIGHTS do not
// depend on the kernel before it, only its activation does.
//
// At decode width these launches are latency-bound — a few hundred blocks that
// each stream a few kilobytes of weights and wait out the DRAM round trip — and in
// a plain stream every one of them starts only after the kernel before it has
// finished and flushed. Launched with programmatic stream serialization, a
// kernel's blocks may start while its predecessor is still running: they
// prefetch exactly the weight bytes they will read into L2, then wait at
// `griddepcontrol.wait` — which returns once every prerequisite grid has
// completed and its writes are visible — and only then read the activation. The
// weights arrive during the predecessor's tail instead of after it.
//
// The CONTRACT, on which correctness rests: a kernel launched with the attribute
// (`launch_pdl`) must call `pdl_wait()` before it reads anything an earlier
// kernel wrote, or touches any buffer an earlier kernel still uses. A kernel
// that calls `pdl_launch_dependents()` lets the next PDL launch start early;
// calling it is harmless when the next launch is an ordinary one.
//
// Below sm_90 every helper compiles to nothing and `launch_pdl` launches
// ordinarily, so the 3090 and the 4090 Mobile run exactly what they ran before.

#include <cuda_runtime.h>
#include <stdint.h>

__device__ __forceinline__ void pdl_launch_dependents() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

__device__ __forceinline__ void pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}

// Prefetch `bytes` from `p` into L2, one 128-byte line per thread-iteration.
__device__ __forceinline__ void pdl_prefetch_l2(const void* p, int bytes, int tid, int nthreads) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    const char* base = reinterpret_cast<const char*>(p);
    for (int off = tid * 128; off < bytes; off += nthreads * 128) {
        asm volatile("prefetch.global.L2 [%0];" :: "l"(base + off));
    }
#endif
}

// Launch `kfn` with programmatic stream serialization where the device supports
// it (compute capability 9.0+), ordinarily otherwise. `kfn` must honour the
// contract above.
static inline cudaError_t launch_pdl(
    const void* kfn, dim3 grid, dim3 block, void** args, size_t smem, cudaStream_t stream)
{
    static int supported = -1;
    if (supported < 0) {
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        supported = major >= 9 ? 1 : 0;
    }
    cudaLaunchAttribute attr;
    attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attr.val.programmaticStreamSerializationAllowed = 1;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = grid;
    cfg.blockDim = block;
    cfg.dynamicSmemBytes = smem;
    cfg.stream = stream;
    cfg.attrs = &attr;
    cfg.numAttrs = supported ? 1 : 0;
    return cudaLaunchKernelExC(&cfg, kfn, args);
}
