// =============================================================================
// kv_record_fill — build KvHead[n_kv_head] records on the device
// =============================================================================
//
// Replaces a host serialize + upload per record. The record body is mostly
// *derivable* from data the device already has, so the host ships a descriptor and
// the kernel writes the bytes:
//
//   - band pointers  — computed as extents[arena].base + chunk * extents[arena].stride
//   - palette maps   — the identity map is arithmetic (2 bits per element); only a
//                      populated map is shipped
//   - outer scales    — default 1.0f; only populated scales are shipped
//   - format tags     — always shipped, but they are one byte per band
//
// Work is flattened across records and grid-strided, so a whole batch is a single
// launch (invariant 2b: the kernel takes a descriptor table, it does not require the
// caller to pack records together) and a small batch still spreads over the machine.
//
//   Grid:  (min(items / 256, 8 * SM), 1, 1)   — see kv_record_fill_blocks
//   Block: (256, 1, 1)
//
// The byte layout MUST match `meta_pool::serialize_kv_heads` (the reference) and
// `paged-decode/slot_types.cuh` (the consumer). Per head, at
// head_base = h * (HD/2 + NP*26):
//
//   k_pal   @ +0                 HD/4 bytes
//   v_pal   @ +HD/4              HD/4 bytes
//   k_ptr[p]@ +HD/2 + p*8        8 bytes each
//   v_ptr[p]@ +HD/2 + NP*8 + p*8
//   k_fmt[p]@ +HD/2 + NP*16 + p  1 byte each
//   v_fmt[p]@ +HD/2 + NP*17 + p
//   k_scl[p]@ +HD/2 + NP*18 + p*4
//   v_scl[p]@ +HD/2 + NP*22 + p*4
//
// A `gid` below 0 is an absent band and writes a null pointer, exactly as the
// reference leaves its zero-initialised slot.

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

// Palette-map density. The identity map names at most this many bands whatever the
// record's band count is, because the map is 2-bit packed — the reference says the
// same thing in `sub_hd`/`N_PALETTE` terms, and the single-latent path never reads
// the map at all.
#define KVREC_N_PALETTE 4

// One record's inputs. Offsets index the packed arrays; -1 means "not supplied,
// derive it", which is what keeps the upload smaller than the records it writes.
struct KvRecordDesc {
    uint64_t dst;       // record's device address
    int64_t gid_off;    // into gids[]: n_kv_head * n_palette * 2 entries
    int64_t pal_off;    // into pal[]:  n_kv_head * (HD/4) * 2 bytes, or -1 = identity
    int64_t fmt_off;    // into fmt[]:  n_kv_head * n_palette * 2 bytes
    int64_t scale_off;  // into scales[]: n_kv_head * n_palette * 2 floats, or -1 = unity
};

// An arena's slot geometry, indexed by arena index. `base == 0` marks an arena that
// is not resident, whose bands resolve to a null pointer.
struct KvArenaExtent {
    uint64_t base;
    int64_t stride;
};

// The host uploads both of these as flat int64 arrays and this side reinterprets them,
// so nothing but these assertions ties the two layouts together. Rust asserts its own
// sizes against DESC_WORDS/EXTENT_WORDS in `kv_record_fill.rs`; these are the other half
// of that pair. Without them, inserting or reordering a field here would have the kernel
// read `gid_off` out of `dst`'s bytes and dereference whatever pointer that forms.
static_assert(sizeof(KvRecordDesc) == 5 * 8, "KvRecordDesc must be 5 int64 words");
static_assert(sizeof(KvArenaExtent) == 2 * 8, "KvArenaExtent must be 2 int64 words");
static_assert(offsetof(KvRecordDesc, dst) == 0, "dst is word 0");
static_assert(offsetof(KvRecordDesc, gid_off) == 8, "gid_off is word 1");
static_assert(offsetof(KvRecordDesc, pal_off) == 16, "pal_off is word 2");
static_assert(offsetof(KvRecordDesc, fmt_off) == 24, "fmt_off is word 3");
static_assert(offsetof(KvRecordDesc, scale_off) == 32, "scale_off is word 4");
static_assert(offsetof(KvArenaExtent, base) == 0, "base is word 0");
static_assert(offsetof(KvArenaExtent, stride) == 8, "stride is word 1");

// K and V arrive as SEPARATE arrays, each laid out exactly as the host already holds
// it (`k_pal[h * pal_bytes ..]`, `k_fmt[h * n_palette + p]`, likewise for V). Keeping
// them separate is what lets the host upload its existing slices verbatim instead of
// interleaving them into one buffer first — which would be precisely the per-band host
// loop this kernel exists to remove.
// Threads per block. 256 is the same choice `kv_ptr_patch` makes for the same reason:
// enough to fill a block's worth of warps at every geometry in the table, small enough
// that several blocks co-reside per SM.
#define KVREC_THREADS 256

// `__launch_bounds__` caps the register budget so the occupancy is the grid's to decide
// rather than the register allocator's. Without it nvcc is free to spill the band loop's
// live values into more registers than a full complement of blocks can afford, and the
// occupancy drops silently — there is no diagnostic, just fewer resident blocks.
extern "C" __global__ __launch_bounds__(KVREC_THREADS) void kv_record_fill_kernel(
    const KvRecordDesc* __restrict__ descs,
    const int64_t* __restrict__ gids,
    const uint8_t* __restrict__ k_pal,
    const uint8_t* __restrict__ v_pal,
    const uint8_t* __restrict__ k_fmt,
    const uint8_t* __restrict__ v_fmt,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const KvArenaExtent* __restrict__ extents,
    int n_extents,
    int n_records,
    int n_kv_head,
    int head_dim,
    int n_palette,
    int gid_stride,   // Rust's GID_STRIDE — passed so the packing has ONE definition
    int invalid_tag)  // ArenaFormatTag::Invalid, for a band whose tag was not recorded
{
    const int pal_bytes = head_dim / 4;
    const int head_sz = head_dim / 2 + n_palette * 26;
    const int sub_hd = (head_dim / KVREC_N_PALETTE) > 0 ? (head_dim / KVREC_N_PALETTE) : 1;
    // GID slice stride per head is the RECORD's band count, not the global
    // per-head width: an 8-band single-latent head reads 16 gids.
    const int gstride = n_palette * 2;

    // Work is flattened across records and **grid-strided**, not one block per record.
    // A batch is often a handful of chunks, and one block per record left all but a
    // handful of SMs idle; flattening means a 4-record batch still spreads over the
    // machine, and a 4,000-record batch does not need 4,000 blocks resident.
    //
    // The two phases are counted separately because their item sizes differ — palette
    // bytes are bytes, bands are a pointer plus a tag plus a scale — and a single
    // flattened space would give one phase's threads the other's stride.
    const int pal_vecs = (pal_bytes * 2) >> 2;          // 4 bytes at a time, per head
    const int pal_tail = (pal_bytes * 2) & 3;           // bytes the vector pass leaves
    const int pal_items = n_kv_head * (pal_vecs + (pal_tail ? 1 : 0));
    const int band_items = n_kv_head * n_palette * 2;
    const int64_t total = (int64_t)n_records * (pal_items + band_items);
    const int64_t step = (int64_t)gridDim.x * blockDim.x;

    for (int64_t t = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += step) {
        const int rec = (int)(t / (pal_items + band_items));
        const int within = (int)(t - (int64_t)rec * (pal_items + band_items));
        const KvRecordDesc d = descs[rec];
        uint8_t* out = reinterpret_cast<uint8_t*>(d.dst);
        if (out == nullptr) continue;

        if (within < pal_items) {
            // ---- palette maps ------------------------------------------------
            // Four bytes per thread through a `uchar4` store. The maps are
            // `2 * pal_bytes` contiguous bytes at the head's base and the record slot
            // is at least 8-byte aligned, so a 4-byte store is aligned whenever
            // `head_sz % 4 == 0` — which the caller's alignment check guarantees.
            const int per_head = pal_vecs + (pal_tail ? 1 : 0);
            const int h = within / per_head;
            const int v_i = within - h * per_head;
            uint8_t* map_out = out + h * head_sz;

            const bool is_tail = (pal_tail != 0) && (v_i == pal_vecs);
            const int base_byte = v_i * 4;
            const int n_bytes = is_tail ? pal_tail : 4;

            uint8_t quad[4];
            for (int b = 0; b < n_bytes; ++b) {
                const int rem = base_byte + b;  // byte within this head's two maps
                const int is_v = rem >= pal_bytes;
                const int byte_in_map = is_v ? rem - pal_bytes : rem;
                if (d.pal_off >= 0) {
                    const uint8_t* src = is_v ? v_pal : k_pal;
                    quad[b] = src[d.pal_off + h * pal_bytes + byte_in_map];
                } else {
                    // Identity: element e routes to band min(e / sub_hd, NP-1), packed
                    // two bits per element, four per byte. Same map for K and V.
                    uint8_t acc = 0;
                    for (int q = 0; q < 4; ++q) {
                        const int elem = byte_in_map * 4 + q;
                        if (elem >= head_dim) break;
                        int idx = elem / sub_hd;
                        if (idx > KVREC_N_PALETTE - 1) idx = KVREC_N_PALETTE - 1;
                        acc |= (uint8_t)(idx) << (q * 2);
                    }
                    quad[b] = acc;
                }
            }
            if (n_bytes == 4) {
                *reinterpret_cast<uchar4*>(map_out + base_byte) =
                    make_uchar4(quad[0], quad[1], quad[2], quad[3]);
            } else {
                for (int b = 0; b < n_bytes; ++b) map_out[base_byte + b] = quad[b];
            }
        } else {
            // ---- pointers, tags and scales -----------------------------------
            // One thread per (head, band, side), each resolving its band's absolute
            // address from the extent table — the work that used to be done on the host
            // and shipped over the bus.
            const int i = within - pal_items;
            const int h = i / (n_palette * 2);
            const int rem = i - h * (n_palette * 2);
            const int p = rem >> 1;
            const int is_v = rem & 1;
            uint8_t* head_out = out + h * head_sz;

            const int64_t raw = gids[d.gid_off + h * gstride + p * 2 + is_v];
            uint64_t addr = 0;
            if (raw >= 0) {
                const int arena = (int)(raw / gid_stride);
                const int64_t chunk = raw % gid_stride;
                if (arena >= 0 && arena < n_extents) {
                    const KvArenaExtent e = extents[arena];
                    if (e.base != 0 && e.stride > 0) {
                        addr = e.base + (uint64_t)(chunk * e.stride);
                    }
                }
            }
            const int ptr_off = head_dim / 2 + (is_v ? n_palette * 8 : 0) + p * 8;
            *reinterpret_cast<uint64_t*>(head_out + ptr_off) = addr;

            // Format tag. Absent ⇒ Invalid, never a float tag: an unrecorded band has
            // no known layout, and naming one would decode quantized bytes as floats.
            const int tag_i = h * n_palette + p;
            const uint8_t* fmt_src = is_v ? v_fmt : k_fmt;
            const uint8_t tag =
                (d.fmt_off >= 0) ? fmt_src[d.fmt_off + tag_i] : (uint8_t)invalid_tag;
            head_out[head_dim / 2 + n_palette * (is_v ? 17 : 16) + p] = tag;

            // Outer scale. Absent ⇒ unity.
            const float* scale_src = is_v ? v_scale : k_scale;
            const float s = (d.scale_off >= 0) ? scale_src[d.scale_off + tag_i] : 1.0f;
            *reinterpret_cast<float*>(head_out + head_dim / 2 + n_palette * (is_v ? 22 : 18)
                                      + p * 4) = s;
        }
    }
}

// Blocks for `n_items` total work items.
//
// Capped rather than one-block-per-item: past a few waves per SM more blocks buy
// nothing but scheduling, and the grid-stride loop above means a capped grid still
// covers any amount of work. The same shape and the same reasoning as
// `kv_ptr_patch_blocks`.
static inline int kv_record_fill_blocks(int64_t n_items) {
    const int64_t by_work = (n_items + KVREC_THREADS - 1) / KVREC_THREADS;
    // Both queries are checked. They cannot fail in practice — a launch is imminent, so
    // there is a current context — but an ignored error would leave `sms` uninitialised
    // and size the grid from a stack value. On failure fall back to a conservative SM
    // count: the grid-stride loop covers the work at any grid size, so a wrong count
    // costs occupancy, never correctness.
    int dev = 0;
    int sms = 0;
    if (cudaGetDevice(&dev) != cudaSuccess ||
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess) {
        sms = 32;
    }
    if (sms <= 0) sms = 32;
    const int64_t cap = (int64_t)sms * 8;  // 8 waves per SM saturates a store-bound pass
    const int64_t n = by_work < cap ? by_work : cap;
    return (int)(n < 1 ? 1 : n);
}

extern "C" void run_kv_record_fill(const void* descs, const int64_t* gids, const uint8_t* k_pal,
                                   const uint8_t* v_pal, const uint8_t* k_fmt,
                                   const uint8_t* v_fmt, const float* k_scale,
                                   const float* v_scale, const void* extents, int n_extents,
                                   int n_records, int n_kv_head, int head_dim, int n_palette,
                                   int gid_stride, int invalid_tag, cudaStream_t stream) {
    if (n_records <= 0) return;
    // **The alignment precondition lives here, with the stores that depend on it.** A
    // band pointer is a `uint64_t` at `head_dim / 2 + p * 8` inside a head, and heads sit
    // `head_sz` apart, so both must be multiples of 8 or the store faults
    // (`CUDA_ERROR_MISALIGNED_ADDRESS`). The caller checks this too, but callers are
    // plural — the microbench and the byte-exact test launch directly — so a check that
    // only one of them performs is a check that can be bypassed. Refusing to launch is
    // better than faulting: the records stay as they were and the error surfaces at the
    // caller's next sync rather than as a dead context.
    const int head_sz_check = head_dim / 2 + n_palette * 26;
    if ((head_dim % 16) != 0 || (head_sz_check % 8) != 0) return;
    // The grid is sized from the total work, not from the record count — see
    // `kv_record_fill_blocks`. Mirrors the host-side item count in the kernel so the two
    // cannot disagree about how much there is to do.
    const int pal_bytes = head_dim / 4;
    const int pal_vecs = (pal_bytes * 2) >> 2;
    const int pal_tail = (pal_bytes * 2) & 3;
    const int64_t items_per_record =
        (int64_t)n_kv_head * (pal_vecs + (pal_tail ? 1 : 0)) + (int64_t)n_kv_head * n_palette * 2;
    dim3 grid(kv_record_fill_blocks((int64_t)n_records * items_per_record));
    dim3 block(KVREC_THREADS);
    kv_record_fill_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const KvRecordDesc*>(descs), gids, k_pal, v_pal, k_fmt, v_fmt, k_scale,
        v_scale, reinterpret_cast<const KvArenaExtent*>(extents), n_extents, n_records, n_kv_head,
        head_dim, n_palette, gid_stride, invalid_tag);
}
