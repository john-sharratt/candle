// =============================================================================
// PLE ROW DEQUANT — gathered Q8_0 records → F32 on the card
// =============================================================================
// The n-gram table's gathered rows travel to the device in their on-disk
// Q8_0 form (170 bytes per 160-wide row: five [f16 scale | 32×i8] blocks) —
// ~2.7 KB per token instead of 10 KB of F32 — and this kernel widens them
// device-side. One thread per output element: the 160-way block reads five
// f16 scales through shared broadcast-friendly access patterns and the int8
// payload coalesced; the whole forward gathers 16 rows per token, so this
// launch is latency-bound and deliberately simple. §0.1's "only the gathered
// result crosses PCIe", with the result now crossing in quantized form.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define PLE_Q8_BLOCK_BYTES 34
#define PLE_ROW_ELEMS 160
#define PLE_ROW_BYTES (5 * PLE_Q8_BLOCK_BYTES)

extern "C" __global__ void ple_dequant_q8_kernel(
    const uint8_t* __restrict__ records, // [n_rows, 170]
    float* __restrict__ out,             // [n_rows, 160]
    int n_rows)
{
    const int row = blockIdx.x;
    const int t = threadIdx.x; // 0..159
    if (row >= n_rows) return;
    const uint8_t* r = records + (size_t)row * PLE_ROW_BYTES;
    const int blk = t >> 5;
    const int i = t & 31;
    const uint8_t* b = r + blk * PLE_Q8_BLOCK_BYTES;
    // Rows are 170-byte strided from an aligned upload, so every block start
    // is 2-byte aligned — a legal __half load.
    const float d = __half2float(*reinterpret_cast<const __half*>(b));
    const int8_t q = static_cast<int8_t>(b[2 + i]);
    out[(size_t)row * PLE_ROW_ELEMS + t] = d * static_cast<float>(q);
}

extern "C" void run_ple_dequant_q8(
    const uint8_t* records,
    float* out,
    int n_rows,
    void* stream)
{
    if (n_rows <= 0) return;
    ple_dequant_q8_kernel<<<n_rows, PLE_ROW_ELEMS, 0, (cudaStream_t)stream>>>(
        records, out, n_rows);
}
