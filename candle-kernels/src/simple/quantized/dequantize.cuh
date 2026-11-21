// The per-format dequantization functions, the k-quants, the KS attention-sink
// formats, and the R16 format.
//
// `dequantize_block` and the `dequantize_block_<format>` functions expand one
// block of each quantized format into values. The `DEQUANTIZE` and
// `DEQUANTIZE_K` macros generate the global kernel wrappers, and the KS
// (attention-sink) and R16 formats each have their own dequantizer.

static __device__ __forceinline__ void dequantize_q4_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_0 * x = (const block_q4_0 *) vx;

    const dfloat d = x[ib].d;

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {8.0f, 8.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 8.0f) * d;
    v.y = (v.y - 8.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q4_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q4_1 * x = (const block_q4_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    const int vui = x[ib].qs[iqs];

    v.x = vui & 0xF;
    v.y = vui >> 4;

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_0 * x = (const block_q5_0 *) vx;

    const dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hsub2(v, {16.0f, 16.0f});
    v = __hmul2(v, {d, d});
#else
    v.x = (v.x - 16.0f) * d;
    v.y = (v.y - 16.0f) * d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q5_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q5_1 * x = (const block_q5_1 *) vx;

    const dfloat d = __low2half(x[ib].dm);
    const dfloat m = __high2half(x[ib].dm);

    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));

    const int xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    const int xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y = ((x[ib].qs[iqs] >>  4) | xh_1);

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
    v = __hadd2(v, {m, m});
#else
    v.x = (v.x * d) + m;
    v.y = (v.y * d) + m;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_0(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_0 * x = (const block_q8_0 *) vx;

    const dfloat d = x[ib].d;

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

static __device__ __forceinline__ void dequantize_q8_1(const void * vx, const int ib, const int iqs, dfloat2 & v){
    const block_q8_1 * x = (const block_q8_1 *) vx;

    const dfloat d = __low2half(x[ib].ds);  // delta is in .x

    v.x = x[ib].qs[iqs + 0];
    v.y = x[ib].qs[iqs + 1];

#ifdef GGML_CUDA_F16
    v = __hmul2(v, {d, d});
#else
    v.x *= d;
    v.y *= d;
#endif // GGML_CUDA_F16
}

// The AWQ `dfloat2` dequantizers went with the matmul path they fed. The KV
// arena's own AWQ dequantize is `dequantize_block_q_awq` below, over the
// `block_q_awq_k128` layout, and is unaffected.


template <int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __device__ void dequantize_block(const void * __restrict__ vx, dst_t * __restrict__ y, const int k) {
    const int i = 2*(blockDim.x*blockIdx.x + threadIdx.x);

    if (i >= k) {
        return;
    }

    const int ib = i/qk; // block index
    const int iqs = (i%qk)/qr; // quant index
    const int iybs = i - i%qk; // y block start index
    const int y_offset = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(vx, ib, iqs, v);

    y[iybs + iqs + 0]        = v.x;
    y[iybs + iqs + y_offset] = v.y;
}

template<typename dst_t>
static __device__ void dequantize_block_q4_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_0 * x = (const block_q4_0 *)vx + ib;
    const float d = __half2float(x->d);
    const float dm = -8*d;

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

// MXFP4 dequant — same 256-elems-per-threadblock / 32-thread layout as q4_0.
template<typename dst_t>
static __device__ void dequantize_block_mxfp4(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int64_t i = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }
    dst_t * y = yy + 256*i + 32*ir + 4*il;
    const block_mxfp4 * x = (const block_mxfp4 *)vx + ib;
    const float d = mxfp4_e8m0_half(x->e);
    const uint8_t * q = x->qs + 4*il;
    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = mxfp4_kvalues[q[l] & 0xF] * d;
        y[l+16] = mxfp4_kvalues[q[l] >>  4] * d;
    }
}

template<typename dst_t>
static __device__ void dequantize_block_q4_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {

    const int64_t i = blockIdx.x;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int64_t ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    const block_q4_1 * x = (const block_q4_1 *)vx + ib;
    const float2 d = __half22float2(x->dm);

    const uint8_t * q = x->qs + 4*il;

    for (int l = 0; l < 4; ++l) {
        y[l+ 0] = d.x * (q[l] & 0xF) + d.y;
        y[l+16] = d.x * (q[l] >>  4) + d.y;
    }
}

//================================== k-quants

template<typename dst_t>
static __device__ void dequantize_block_q2_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i   = blockIdx.x;
    const block_q2_K * x = (const block_q2_K *) vx;

    const int tid = threadIdx.x;
#if QK_K == 256
    const int n   = tid/32;
    const int l   = tid - 32*n;
    const int is  = 8*n + l/16;

    const uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#else
    const int is = tid/16;  // 0 or 1
    const int il = tid%16;  // 0...15
    const uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;
    float dall = __low2half(x[i].dm);
    float dmin = __high2half(x[i].dm);
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
static __device__ void dequantize_block_q3_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {

    const int i = blockIdx.x;
    const block_q3_K * x = (const block_q3_K *) vx;

#if QK_K == 256
    const int r = threadIdx.x/4;
    const int tid = r/2;
    const int is0 = r%2;
    const int l0 = 16*is0 + 4*(threadIdx.x%4);
    const int n = tid / 4;
    const int j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int is = 8*n + 2*j + is0;
    int shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    float d_all = x[i].d;
    float dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    const uint8_t * q = x[i].qs + 32*n;
    const uint8_t * hm = x[i].hmask;

    for (int l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#else
    const int tid = threadIdx.x;
    const int is  = tid/16;  // 0 or 1
    const int il  = tid%16;  // 0...15
    const int im  = il/8;    // 0...1
    const int in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    const uint8_t q = x[i].qs[il] >> (2*is);
    const uint8_t h = x[i].hmask[in] >> (2*is + im);
    const float   d = (float)x[i].d;

    if (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } else {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

#if QK_K == 256
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

template<typename dst_t>
static __device__ void dequantize_block_q4_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q4_K * x = (const block_q4_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int is  = 2*il;
    const int n   = 4;

    dst_t * y = yy + i*QK_K + 64*il + n*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * q = x[i].qs + 32*il + n*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = d1 * (q[l] & 0xF) - m1;
        y[l +32] = d2 * (q[l] >>  4) - m2;
    }
#else
    const int tid = threadIdx.x;
    const uint8_t * q = x[i].qs;
    dst_t * y = yy + i*QK_K;
    const float d = (float)x[i].dm[0];
    const float m = (float)x[i].dm[1];
    y[tid+ 0] = d * (x[i].scales[0] & 0xF) * (q[tid] & 0xF) - m * (x[i].scales[0] >> 4);
    y[tid+32] = d * (x[i].scales[1] & 0xF) * (q[tid] >>  4) - m * (x[i].scales[1] >> 4);
#endif
}

template<typename dst_t>
static __device__ void dequantize_block_q5_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q5_K * x = (const block_q5_K *) vx;

    const int i = blockIdx.x;

#if QK_K == 256
    // assume 64 threads - this is very slightly better than the one below
    const int tid = threadIdx.x;
    const int il  = tid/16;   // il is in 0...3
    const int ir  = tid%16;   // ir is in 0...15
    const int is  = 2*il;     // is is in 0...6

    dst_t * y = yy + i*QK_K + 64*il + 2*ir;

    const float dall = __low2half(x[i].dm);
    const float dmin = __high2half(x[i].dm);

    const uint8_t * ql = x[i].qs + 32*il + 2*ir;
    const uint8_t * qh = x[i].qh + 2*ir;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, x[i].scales, sc, m);
    const float d1 = dall * sc; const float m1 = dmin * m;
    get_scale_min_k4(is + 1, x[i].scales, sc, m);
    const float d2 = dall * sc; const float m2 = dmin * m;

    uint8_t   hm  = 1 << (2*il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#else
    const int tid = threadIdx.x;
    const uint8_t q = x[i].qs[tid];
    const int im = tid/8;  // 0...3
    const int in = tid%8;  // 0...7
    const int is = tid/16; // 0 or 1
    const uint8_t h = x[i].qh[in] >> im;
    const float d = x[i].d;
    dst_t * y = yy + i*QK_K + tid;
    y[ 0] = d * x[i].scales[is+0] * ((q & 0xF) - ((h >> 0) & 1 ? 0 : 16));
    y[32] = d * x[i].scales[is+2] * ((q >>  4) - ((h >> 4) & 1 ? 0 : 16));
#endif
}

template<typename dst_t>
static __device__ void dequantize_block_q6_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q6_K * x = (const block_q6_K *) vx;

    const int64_t i = blockIdx.x;
#if QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/32;   // ip is 0 or 1
    const int64_t il  = tid - 32*ip; // 0...32
    const int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    const float d = x[i].d;

    const uint8_t * ql = x[i].ql + 64*ip + il;
    const uint8_t   qh = x[i].qh[32*ip + il];
    const int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#else

    // assume 32 threads
    const int64_t tid = threadIdx.x;
    const int64_t ip  = tid/16;         // 0 or 1
    const int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    const float d = x[i].d;

    const uint8_t   ql = x[i].ql[16*ip + il];
    const uint8_t   qh = x[i].qh[il] >> (2*ip);
    const int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

template<typename dst_t>
static __device__ void dequantize_block_q8_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i = blockIdx.x;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 8*il;

    const block_q8_0 * x = (const block_q8_0 *)vx + ib;
    const float d = __half2float(x->d);

    const int8_t * q = x->qs + 8*il;

    for (int l = 0; l < 8; ++l) {
        y[l] = d * q[l];
    }
}

// Q8_1 dequantization - similar to Q8_0 but uses ds.x for delta
template<typename dst_t>
static __device__ void dequantize_block_q8_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i = blockIdx.x;

    // assume 32 threads
    const int tid = threadIdx.x;
    const int il  = tid/8;
    const int ir  = tid%8;
    const int ib = 8*i + ir;
    if (ib >= nb32) {
        return;
    }

    dst_t * y = yy + 256*i + 32*ir + 8*il;

    const block_q8_1 * x = (const block_q8_1 *)vx + ib;
    const float d = __low2float(x->ds);  // ds.x = delta

    const int8_t * q = x->qs + 8*il;

    for (int l = 0; l < 8; ++l) {
        y[l] = d * q[l];
    }
}

// =============================================================================
// KS (attention-sink sub-block) block types and dequantization kernels
// Block types must match blocks.cuh exactly.
// =============================================================================

#define QK_Q4_KS 32
typedef struct { half d; uint8_t sa; uint8_t sb; uint8_t qs[16]; } block_q4_ks;

#define QK_Q8_KS 32
typedef struct { half d; uint8_t sa; uint8_t sb; int8_t  qs[32]; } block_q8_ks;

// Q4_KS dequantization
// Layout: qs[k] = low_nibble(k) | (high_nibble(k+16) << 4), biased +8
// Sub-block A: elements 0-3 (attention sinks), sub-block B: elements 4-31
// actual_d_A = d * sa / 255.0,  actual_d_B = d * sb / 255.0
template<typename dst_t>
static __device__ void dequantize_block_q4_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid / 8;    // sub-group index [0,3] — 4 groups of 8 threads, each outputting 4 elements
    const int ir  = tid % 8;    // within-group index [0,7] — selects which of 8 blocks in this blockIdx

    const int ib = 8 * i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256 * i + 32 * ir + 4 * il;

    const block_q4_ks * x = (const block_q4_ks *)vx + ib;
    const float cd = __half2float(x->d);
    const float da = cd * (x->sa / 255.0f);
    const float db = cd * (x->sb / 255.0f);

    const uint8_t * q = x->qs + 4 * il;

    for (int l = 0; l < 4; ++l) {
        // Low nibble: elements 0-15. Scale by da if il==0 (elems 0-3 are sub-block A).
        const float d_lo = (il == 0) ? da : db;
        y[l]      = (dst_t)(d_lo * (float)((int)(q[l] & 0xF) - 8));
        // High nibble: elements 16-31. Always sub-block B scale.
        y[l + 16] = (dst_t)(db   * (float)((int)(q[l] >> 4)  - 8));
    }
}

// Q8_KS dequantization
// Sub-block A: elements 0-3 (il==0, l<4), sub-block B: remainder
template<typename dst_t>
static __device__ void dequantize_block_q8_ks(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid / 8;
    const int ir  = tid % 8;

    const int ib = 8 * i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256 * i + 32 * ir + 8 * il;

    const block_q8_ks * x = (const block_q8_ks *)vx + ib;
    const float cd = __half2float(x->d);
    const float da = cd * (x->sa / 255.0f);
    const float db = cd * (x->sb / 255.0f);

    const int8_t * q = x->qs + 8 * il;

    for (int l = 0; l < 8; ++l) {
        const float d = (il == 0 && l < 4) ? da : db;
        y[l] = (dst_t)(d * (float)q[l]);
    }
}

// Q8_K dequantization - K-quant 8-bit (256 elements per block)
template<typename dst_t>
static __device__ void dequantize_block_q8_K(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q8_K * x = (const block_q8_K *) vx;

    const int64_t i = blockIdx.x;

    // assume 64 threads - 4 elements per thread, 256 elements total
    const int64_t tid = threadIdx.x;
    const int64_t il = tid * 4;  // each thread handles 4 consecutive elements

    dst_t * y = yy + i * QK_K + il;

    const float d = x[i].d;
    const int8_t * q = x[i].qs + il;

    y[0] = d * q[0];
    y[1] = d * q[1];
    y[2] = d * q[2];
    y[3] = d * q[3];
}

template<typename dst_t>
static __device__ void dequantize_block_q5_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block<QK5_0, QR5_0, dequantize_q5_0>(vx, yy, nb32);
}

template<typename dst_t>
static __device__ void dequantize_block_q5_1(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
  return dequantize_block<QK5_1, QR5_1, dequantize_q5_1>(vx, yy, nb32);
}

// =============================================================================
// Q2_0 and Q3_0: simple 2-bit and 3-bit symmetric quantization formats
// Block types must match blocks.cuh exactly.
// =============================================================================

// Q2_0: 2-bit symmetric, scale only (10 bytes per 32 elements)
#define QK2_0 32
typedef struct { half d; uint8_t qs[8]; } block_q2_0;

// Q3_0: 3-bit symmetric, scale only (14 bytes per 32 elements)
#define QK3_0 32
typedef struct { half d; uint8_t qh[4]; uint8_t qs[8]; } block_q3_0;

// Q2_0 dequantization
// Layout: qs[j] = q0|(q1<<2)|(q2<<4)|(q3<<6), levels 0..3
// Value: d * (q - 1.5)  where d = amax/1.5
template<typename dst_t>
static __device__ void dequantize_block_q2_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid / 8;    // 0..3: which 8-element group within the 32-element block
    const int ir  = tid % 8;    // 0..7: which of 8 blocks in this CTA

    const int ib = 8 * i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256 * i + 32 * ir + 8 * il;

    const block_q2_0 * x = (const block_q2_0 *)vx + ib;
    const float d  = __half2float(x->d);
    const float dm = -1.5f * d;  // decode: d*(q - 1.5) = d*q + dm

    // il=0: qs[0..1] => elements 0..7
    // il=1: qs[2..3] => elements 8..15
    // il=2: qs[4..5] => elements 16..23
    // il=3: qs[6..7] => elements 24..31
    const uint8_t * q = x->qs + 2 * il;
    for (int b = 0; b < 2; ++b) {
        uint8_t byte = q[b];
        for (int bit = 0; bit < 4; ++bit) {
            y[4 * b + bit] = (dst_t)(__fmaf_rn(d, (float)((byte >> (2 * bit)) & 3), dm));
        }
    }
}

// Q3_0 dequantization
// Layout: qs[j] = lo0|(lo1<<2)|(lo2<<4)|(lo3<<6), qh[j] = high bits
// Value: d * ((hi<<2|lo) - 3.5)  where d = amax/3.5
template<typename dst_t>
static __device__ void dequantize_block_q3_0(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid / 8;    // 0..3: which 8-element group within the 32-element block
    const int ir  = tid % 8;    // 0..7: which of 8 blocks in this CTA

    const int ib = 8 * i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256 * i + 32 * ir + 8 * il;

    const block_q3_0 * x = (const block_q3_0 *)vx + ib;
    const float d  = __half2float(x->d);
    const float dm = -3.5f * d;  // decode: d*(q - 3.5) = d*q + dm

    const uint8_t * qs = x->qs + 2 * il;
    for (int b = 0; b < 2; ++b) {
        uint8_t byte_qs = qs[b];
        for (int bit = 0; bit < 4; ++bit) {
            int e    = 8 * il + 4 * b + bit;  // element index within block
            int lo   = (byte_qs >> (2 * bit)) & 3;
            int hi   = (x->qh[e >> 3] >> (e & 7)) & 1;
            float qf = (float)((hi << 2) | lo);
            y[4 * b + bit] = (dst_t)(__fmaf_rn(d, qf, dm));
        }
    }
}


// R16 dequantization: extract d[] (K values stored as F16) from block_r16
// block_r16 layout: half d[32] (K), uint16_t q[32] (Q) = 128 bytes total
// Only extracts the K (d[]) values; Q capture data is discarded.
#define QK_R16 32
typedef struct { half d[QK_R16]; uint16_t q[QK_R16]; } block_r16_simple;

template<typename dst_t>
static __device__ void dequantize_block_r16(const void * __restrict__ vx, dst_t * __restrict__ yy, int nb32) {
    const int i   = blockIdx.x;
    const int tid = threadIdx.x;
    const int il  = tid / 8;    // 0..3: which 8-element group within the 32-element block
    const int ir  = tid % 8;    // 0..7: which of 8 blocks in this CTA

    const int ib = 8 * i + ir;
    if (ib >= nb32) return;

    dst_t * y = yy + 256 * i + 32 * ir + 8 * il;

    const block_r16_simple * x = (const block_r16_simple *)vx + ib;
    const half * d = x->d + 8 * il;

    for (int l = 0; l < 8; ++l) {
        y[l] = (dst_t)(__half2float(d[l]));
    }
}

#define DEQUANTIZE_K(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_f32(const void * __restrict__ vx, float * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_f16(const void * __restrict__ vx, half * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_bf16(const void * __restrict__ vx, __nv_bfloat16 * __restrict__ y) { \
  dequantize_block_##QNAME(vx, y); \
} \

#define DEQUANTIZE(QNAME) \
extern "C" __global__ void dequantize_block_##QNAME##_f32(const void * __restrict__ vx, float * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_f16(const void * __restrict__ vx, half * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
} \
extern "C" __global__ void dequantize_block_##QNAME##_bf16(const void * __restrict__ vx, __nv_bfloat16 * __restrict__ y, const int k) { \
  dequantize_block_##QNAME(vx, y, k); \
} \

DEQUANTIZE_K(q2_K)
DEQUANTIZE_K(q3_K)
DEQUANTIZE_K(q4_K)
DEQUANTIZE_K(q5_K)
DEQUANTIZE_K(q6_K)
DEQUANTIZE_K(q8_K)
DEQUANTIZE(q4_0)
DEQUANTIZE(q4_1)
DEQUANTIZE(q5_0)
DEQUANTIZE(q5_1)
DEQUANTIZE(q8_0)
DEQUANTIZE(q8_1)
DEQUANTIZE(mxfp4)
DEQUANTIZE(q4_ks)
DEQUANTIZE(q8_ks)
DEQUANTIZE(q2_0)
DEQUANTIZE(q3_0)
DEQUANTIZE(r16)

// =============================================================================
// K/128 NATIVE TYPES - Dequantization kernels for AWQ quant formats
// =============================================================================
// These types use 128 elements per block (K/128 format) and are designed for
// the GEMX tensor-core matmul path. They can also be dequantized to f32/f16/bf16.
//
// Block types match canonical definitions in block_compact.cuh:
// - Q_AWQ:    80B (64B qs + scale/zero + padding) - asymmetric 4-bit g128
// - Q_AWQ_G64:80B (64B qs + 2×scale/zero) - asymmetric 4-bit g64
// =============================================================================

// Q_AWQ: 16 threads × 8 elements, 1 scale/zero pair per 128 elements
struct __align__(16) block_q_awq_k128 {
    int qs[16];       // 16 ints = 64 bytes
    half scale;       // scale for entire block
    half zero;        // zero point for entire block
    int _pad[3];      // 12 bytes padding to reach 80 bytes
};

// Q_AWQ_G64: 2 groups of 64 elements, each with own scale/zero
struct __align__(16) block_q_awq_g64_k128 {
    int qs[16];       // 16 ints = 64 bytes
    half scales[2];   // scale per 64-element group
    half zeros[2];    // zero per 64-element group
    int _pad[2];      // 8 bytes padding to reach 80 bytes
};

// Q_AWQ: Asymmetric 4-bit with scale and zero (g128)
// value = scale * (nibble - zero)
// Using int[16] layout
template<typename dst_t>
static __device__ void dequantize_block_q_awq(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q_awq_k128 * x = (const block_q_awq_k128 *) vx;
    const int i = blockIdx.x;
    const int tid = threadIdx.x;  // 0-31
    
    dst_t * y = yy + i * 128;
    const float scale = __half2float(x[i].scale);
    const float zero = __half2float(x[i].zero);
    
    // Each thread processes 4 elements (half an int = 4 nibbles)
    const int int_idx = tid / 2;
    const int q = x[i].qs[int_idx];
    const int half_select = tid & 1;
    
    const int shift = half_select * 16;
    const int nibble0 = (q >> (shift + 0)) & 0xF;
    const int nibble1 = (q >> (shift + 4)) & 0xF;
    const int nibble2 = (q >> (shift + 8)) & 0xF;
    const int nibble3 = (q >> (shift + 12)) & 0xF;
    
    const int out_base = tid * 4;
    y[out_base + 0] = (dst_t)(scale * (nibble0 - zero));
    y[out_base + 1] = (dst_t)(scale * (nibble1 - zero));
    y[out_base + 2] = (dst_t)(scale * (nibble2 - zero));
    y[out_base + 3] = (dst_t)(scale * (nibble3 - zero));
}

// Q_AWQ_G64: Asymmetric 4-bit with 2 groups of 64 elements each
// Using int[16] layout, threads 0-15 handle group 0, threads 16-31 handle group 1
template<typename dst_t>
static __device__ void dequantize_block_q_awq_g64(const void * __restrict__ vx, dst_t * __restrict__ yy) {
    const block_q_awq_g64_k128 * x = (const block_q_awq_g64_k128 *) vx;
    const int i = blockIdx.x;
    const int tid = threadIdx.x;  // 0-31
    
    dst_t * y = yy + i * 128;
    
    // First 16 threads handle group 0 (elements 0-63), last 16 handle group 1 (64-127)
    const int group = tid / 16;
    const int local_tid = tid % 16;
    
    const float scale = __half2float(x[i].scales[group]);
    const float zero = __half2float(x[i].zeros[group]);
    
    // Each thread in the group processes 4 elements
    // Group 0: ints 0-7, Group 1: ints 8-15
    const int int_idx = group * 8 + local_tid / 2;
    const int q = x[i].qs[int_idx];
    const int half_select = local_tid & 1;
    
    const int shift = half_select * 16;
    const int nibble0 = (q >> (shift + 0)) & 0xF;
    const int nibble1 = (q >> (shift + 4)) & 0xF;
    const int nibble2 = (q >> (shift + 8)) & 0xF;
    const int nibble3 = (q >> (shift + 12)) & 0xF;
    
    const int out_base = group * 64 + local_tid * 4;
    y[out_base + 0] = (dst_t)(scale * (nibble0 - zero));
    y[out_base + 1] = (dst_t)(scale * (nibble1 - zero));
    y[out_base + 2] = (dst_t)(scale * (nibble2 - zero));
    y[out_base + 3] = (dst_t)(scale * (nibble3 - zero));
}

// Generate global kernel wrappers for K/128 AWQ types
DEQUANTIZE_K(q_awq)
DEQUANTIZE_K(q_awq_g64)
