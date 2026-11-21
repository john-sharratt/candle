// The Q0_V and KV block encode and decode oracles and their round-trip entry
// points.
//
// The `q0_v_decode_oracle_kernel` and `q0_v_encode_oracle_kernel` for K-side
// and V-side Q0_V blocks, the `kv_decode_oracle_kernel` and
// `kv_encode_oracle_kernel` for every arena block format, the
// `roundtrip_kernel_body` with its `roundtrip_q0_v_*` entry points, and
// `roundtrip_q0_v_runtime_kernel`, the runtime-table round trip used for curve
// selection, each with its `run_*` launcher.

// =============================================================================
// Q0_V DECODE ORACLE  (K-side and V-side)
// =============================================================================
// Decodes every element of `num_blocks` Q0_V blocks through each production
// read path that reaches a Q0_V block, under side `is_k`'s codebook, at
// palette scale `scale`:
//   via_dispatch — `dequant_element_inline`, the ArenaAccessor's element path
//   via_i8_elem  — `int8_elem::i8_dequant_elem`, the INT8 prefill kernel's path
//   via_quad     — `int8_elem::I8BlockQuad<block_q0_v>::load4`, the INT8 tile
//                  decode kernel's four-token path, handed `1 / scale` as its
//                  callers compute it
//   via_header   — `q0_v_header` + `q0_v_header_elem`, the hoisted header the
//                  decode kernels hold per block
// The Rust decode-oracle test compares all four, bit for bit, against the
// reference `k_quants::q0_v_elem_scaled` over every 16-bit code. One thread
// per four-token quad: `num_blocks * 8` threads.
template <bool IS_K>
__global__ void q0_v_decode_oracle_kernel(
    const block_q0_v* __restrict__ src,
    float* __restrict__ via_dispatch,
    float* __restrict__ via_i8_elem,
    float* __restrict__ via_quad,
    float* __restrict__ via_header,
    int num_blocks,
    float scale)
{
    const int q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= num_blocks * 8) return;
    const int blk = q >> 3;
    const int t = (q & 7) * 4;
    const block_q0_v* b = src + blk;
    const float r = 1.f / scale;
    float o[4];
    int8_elem::I8BlockQuad<block_q0_v, IS_K>::load4(b, t, r, o);
    const Q0VHeader h = q0_v_header<IS_K>(b, r);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int e = t + j;
        const size_t i = (size_t)blk * QK_Q0_V + e;
        via_quad[i] = o[j];
        via_header[i] = q0_v_header_elem(h, e);
        via_dispatch[i] = dequant_element_inline<float, IS_K>(b, e, ArenaFormat::Q0_V, scale);
        via_i8_elem[i] = int8_elem::i8_dequant_elem<IS_K>(
            ArenaFormat::Q0_V, reinterpret_cast<const char*>(b), e, scale);
    }
}

extern "C" void run_q0_v_decode_oracle(
    const void* src, void* via_dispatch, void* via_i8_elem, void* via_quad, void* via_header,
    int num_blocks, int is_k, float scale, void* stream)
{
    if (num_blocks <= 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    const int threads = 256;
    const int blocks = (num_blocks * 8 + threads - 1) / threads;
    if (is_k) {
        q0_v_decode_oracle_kernel<true><<<blocks, threads, 0, s>>>(
            (const block_q0_v*)src, (float*)via_dispatch, (float*)via_i8_elem,
            (float*)via_quad, (float*)via_header, num_blocks, scale);
    } else {
        q0_v_decode_oracle_kernel<false><<<blocks, threads, 0, s>>>(
            (const block_q0_v*)src, (float*)via_dispatch, (float*)via_i8_elem,
            (float*)via_quad, (float*)via_header, num_blocks, scale);
    }
}

// =============================================================================
// Q0_V ENCODE ORACLE  (K-side and V-side)
// =============================================================================
// Encodes `num_blocks` 32-element blocks of `src` (f32, already outer-scaled)
// with the production per-block encoder `quantize_blocks_q0_v<IS_K>`, one warp
// per block. The Rust encode-oracle test compares the bytes with the
// reference encoder `k_quants::encode_block_q0_v`.
template <bool IS_K>
__global__ void q0_v_encode_oracle_kernel(
    const float* __restrict__ src, block_q0_v* __restrict__ dst, int num_blocks)
{
    quantize_blocks_q0_v<IS_K, 1>(src, dst, num_blocks);
}

extern "C" void run_q0_v_encode_oracle(const void* src, void* dst, int num_blocks, int is_k, void* stream)
{
    if (num_blocks <= 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    const int threads = 256;
    const int blocks = (num_blocks + threads / 32 - 1) / (threads / 32);
    if (is_k) {
        q0_v_encode_oracle_kernel<true><<<blocks, threads, 0, s>>>(
            (const float*)src, (block_q0_v*)dst, num_blocks);
    } else {
        q0_v_encode_oracle_kernel<false><<<blocks, threads, 0, s>>>(
            (const float*)src, (block_q0_v*)dst, num_blocks);
    }
}

// =============================================================================
// KV BLOCK DECODE / ENCODE ORACLES  (every arena block format)
// =============================================================================
// The decode oracle reads every element of `num_blocks` blocks of arena format
// `fmt` (`block_bytes` apart) through `dequant_element_inline` — the element
// path every paged attention kernel's ArenaAccessor takes — at unit palette
// scale, under side `is_k`. The encode oracle encodes `num_blocks` 32-element
// f32 blocks with `p4c_encode_quant_block`, the encoder the palette seal writes
// every sealed band with, one warp per block. The Rust oracle tests compare
// both with the host block codecs, bit for bit and byte for byte.
template <bool IS_K>
__global__ void kv_decode_oracle_kernel(
    const char* __restrict__ src, float* __restrict__ dst,
    int num_blocks, int block_bytes, int fmt)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_blocks * 32) return;
    const int blk = i >> 5;
    dst[i] = dequant_element_inline<float, IS_K>(
        src + (size_t)blk * block_bytes, i & 31, fmt, 1.0f);
}

extern "C" void run_kv_decode_oracle(
    const void* src, void* dst, int num_blocks, int block_bytes, int fmt, int is_k, void* stream)
{
    if (num_blocks <= 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    const int threads = 256;
    const int blocks = (num_blocks * 32 + threads - 1) / threads;
    if (is_k) {
        kv_decode_oracle_kernel<true><<<blocks, threads, 0, s>>>(
            (const char*)src, (float*)dst, num_blocks, block_bytes, fmt);
    } else {
        kv_decode_oracle_kernel<false><<<blocks, threads, 0, s>>>(
            (const char*)src, (float*)dst, num_blocks, block_bytes, fmt);
    }
}

template <bool IS_K>
__global__ void kv_encode_oracle_kernel(
    const float* __restrict__ src, char* __restrict__ dst,
    int num_blocks, int block_bytes, int fmt)
{
    const int blk = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (blk >= num_blocks) return;
    p4c_encode_quant_block<IS_K>(src + (size_t)blk * 32, dst + (size_t)blk * block_bytes, fmt);
}

extern "C" void run_kv_encode_oracle(
    const void* src, void* dst, int num_blocks, int block_bytes, int fmt, int is_k, void* stream)
{
    if (num_blocks <= 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    const int threads = 256;
    const int blocks = (num_blocks + threads / 32 - 1) / (threads / 32);
    if (is_k) {
        kv_encode_oracle_kernel<true><<<blocks, threads, 0, s>>>(
            (const float*)src, (char*)dst, num_blocks, block_bytes, fmt);
    } else {
        kv_encode_oracle_kernel<false><<<blocks, threads, 0, s>>>(
            (const float*)src, (char*)dst, num_blocks, block_bytes, fmt);
    }
}

// =============================================================================
// Q0_V ROUND-TRIP TEST ENTRYPOINTS  (K-side and V-side)
// =============================================================================
// These wrap the *actual* production quantize and dequantize paths used by
// the format-selection kernel. Used by the offline modelling/diagnostic
// test to measure round-trip error per block under the same code paths the
// kernel uses, isolating whether observed Q0_V usage gaps are due to the
// quant kernels themselves or the selection wiring.
//
// Layout:
//   - One warp per 32-element block (lane = element index)
//   - Encoder: warp-cooperative `quantize_block_q0_v_core<IS_K>`
//   - Decoder: warp-cooperative load via `q0_v_load_element_f32<IS_K>`
//   - Pre-scale: caller passes input pre-scaled by `outer` (i.e.
//     `src * outer`), encoder writes the resulting block.
//   - Post-scale: decoder divides by `outer` to recover original units;
//     for our metric work we pass `outer = 1.0` so the round-trip is in
//     the encoder's natural normalised space.

template <bool IS_K>
__device__ __forceinline__ void roundtrip_kernel_body(
    const float* __restrict__ src, float* __restrict__ recon,
    int blk, int num_blocks, int lane, float outer)
{
    if (blk >= num_blocks) return;
    const float xi = src[blk * QK_Q0_V + lane] * outer;
    block_q0_v packed;
    quantize_block_q0_v_core<IS_K>(xi, &packed);
    // All 32 lanes share the just-encoded block; broadcast lane 0's bytes
    // to every lane (warp-uniform after the encoder's lane-0 store).
    const unsigned lo_bits = __shfl_sync(0xffffffff, (unsigned)packed.lo, 0, 32);
    const unsigned hi_bits = __shfl_sync(0xffffffff, (unsigned)packed.hi, 0, 32);
    block_q0_v shared_packed;
    shared_packed.lo = (uint8_t)lo_bits;
    shared_packed.hi = (uint8_t)hi_bits;
    recon[blk * QK_Q0_V + lane] =
        q0_v_load_element_f32<IS_K>(&shared_packed, lane, outer);
}

extern "C" __global__ void roundtrip_q0_v_k_kernel(
    const float* __restrict__ src, float* __restrict__ recon,
    int num_blocks, float outer)
{
    const int warps_per_grid_block = blockDim.x / WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int blk = blockIdx.x * warps_per_grid_block + warp_in_block;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    roundtrip_kernel_body<true>(src, recon, blk, num_blocks, lane, outer);
}

extern "C" __global__ void roundtrip_q0_v_v_kernel(
    const float* __restrict__ src, float* __restrict__ recon,
    int num_blocks, float outer)
{
    const int warps_per_grid_block = blockDim.x / WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int blk = blockIdx.x * warps_per_grid_block + warp_in_block;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    roundtrip_kernel_body<false>(src, recon, blk, num_blocks, lane, outer);
}

extern "C" void run_roundtrip_q0_v(
    const void* src, void* recon, int num_blocks, float outer, int is_k, void* stream)
{
    if (num_blocks <= 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    const int threads = 32;
    const int blocks = num_blocks;
    if (is_k) {
        roundtrip_q0_v_k_kernel<<<blocks, threads, 0, s>>>(
            (const float*)src, (float*)recon, num_blocks, outer);
    } else {
        roundtrip_q0_v_v_kernel<<<blocks, threads, 0, s>>>(
            (const float*)src, (float*)recon, num_blocks, outer);
    }
}

// =============================================================================
// Q0_V RUNTIME-TABLE ROUND-TRIP — diagnostic / curve-selection entry point
// =============================================================================
// Same warp-cooperative encoder + decoder pair the production path uses, but
// the codebook tables (curve, scale, centroid, peak permutation, peak bin
// offsets) are supplied as device pointers at launch time instead of read
// from `__constant__` memory. Used by the iterative curve-selection
// diagnostic to swap the codebook between iterations without recompiling.
//
// Caller layout contract (all on the device):
//   curve_table_flat       : [256][32] i8                    (8192 B)
//   scale_table_bits       : [32]      uint16 (f16 bits)     (   64 B)
//   centroid_table_bits    : [32][8]   uint16 (f16 bits)     (  512 B)
//   peak_curve_indices     : [256]     uint8                 (  256 B)
//   peak_bin_offsets       : [33]      uint16                (   66 B)
//
// Behaviour matches the static path byte-for-byte when the runtime tables
// are populated with the same content as the static `_k`/`_v` arrays.

extern "C" __global__ void roundtrip_q0_v_runtime_kernel(
    const float* __restrict__ src, float* __restrict__ recon,
    int num_blocks, float outer,
    const int8_t*   __restrict__ curve_table_flat,
    const uint16_t* __restrict__ scale_table_bits,
    const uint16_t* __restrict__ centroid_table_bits_flat,
    const uint8_t*  __restrict__ peak_curve_indices,
    const uint16_t* __restrict__ peak_bin_offsets)
{
    const int warps_per_grid_block = blockDim.x / WARP_SIZE;
    const int warp_in_block = threadIdx.x / WARP_SIZE;
    const int blk = blockIdx.x * warps_per_grid_block + warp_in_block;
    const int lane = threadIdx.x & (WARP_SIZE - 1);
    if (blk >= num_blocks) return;

    q0_v_detail::Q0VTablesRuntime tbl{
        curve_table_flat,
        scale_table_bits,
        centroid_table_bits_flat,
        peak_curve_indices,
        peak_bin_offsets,
    };

    const float xi = src[blk * QK_Q0_V + lane] * outer;
    block_q0_v packed;
    quantize_block_q0_v_core_runtime(xi, &packed, tbl);

    // Broadcast the lane-0 packed bytes warp-wide before decode.
    const unsigned lo_bits = __shfl_sync(0xffffffff, (unsigned)packed.lo, 0, 32);
    const unsigned hi_bits = __shfl_sync(0xffffffff, (unsigned)packed.hi, 0, 32);
    block_q0_v shared_packed;
    shared_packed.lo = (uint8_t)lo_bits;
    shared_packed.hi = (uint8_t)hi_bits;

    const float decoded = q0_v_elem_runtime(&shared_packed, lane, tbl);
    recon[blk * QK_Q0_V + lane] = decoded / outer;
}

extern "C" void run_roundtrip_q0_v_runtime(
    const void* src,
    void*       recon,
    int         num_blocks,
    float       outer,
    const void* curve_table_flat,         // [256][32] i8
    const void* scale_table_bits,          // [32]      u16
    const void* centroid_table_bits_flat,  // [32][8]   u16
    const void* peak_curve_indices,        // [256]     u8
    const void* peak_bin_offsets,          // [33]      u16
    void*       stream)
{
    if (num_blocks <= 0) return;
    cudaStream_t s = (cudaStream_t)stream;
    const int threads = 32;
    const int blocks = num_blocks;
    roundtrip_q0_v_runtime_kernel<<<blocks, threads, 0, s>>>(
        (const float*)src, (float*)recon, num_blocks, outer,
        (const int8_t*)curve_table_flat,
        (const uint16_t*)scale_table_bits,
        (const uint16_t*)centroid_table_bits_flat,
        (const uint8_t*)peak_curve_indices,
        (const uint16_t*)peak_bin_offsets);
}
