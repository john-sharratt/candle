#pragma once
// ============================================================================
// PalIter — palette-aware dimension iterator (warp-ballot optimized).
//
// Maps this lane's VEC logical dims (lane*VEC .. lane*VEC+VEC-1) to their
// palette-space smem offsets using warp-collective ballot + popcount.
//
// ArenaAccessor loads each palette into a contiguous smem region of SUB
// elements in ascending **dim** order:  palette p occupies
// [p*SUB .. p*SUB+SUB).  Within that region, dims are stored in the
// order they appear when iterating d = 0..HEAD_DIM-1.
//
// For dim d = lane*VEC+j assigned to palette p, the smem offset is:
//     p * SUB + #{d' < d : palette(d') == p}
//
// Because dims are lane-major (lane 0 owns the smallest VEC dims, lane 1
// the next VEC, etc.), the count decomposes into:
//   (a) ALL VEC slots of each lane' < lane whose palette matches p, plus
//   (b) slots j' < j of this lane whose palette matches p.
//
// Part (a) is computed via warp ballots (one pair per slot jj), masking
// with lane_mask (lanes < me) and summing across all jj.
// Part (b) is a lane-local palette comparison, no warp ops needed.
//
// Cost: 8 ballots, VEC*N_PALETTE=16 popcs, ~10 regs.  Zero divergence.
// operator[](j) is a single register read.
//
// For HD=128 VEC=4: max idx = 3*32+31 = 127, fits in uint8_t.
// ============================================================================
template <int VEC, int HEAD_DIM>
struct PalIter {
    static constexpr int SUB = HEAD_DIM / N_PALETTE;
    uint8_t scatter[VEC];

    __device__ __forceinline__ void init(const uint8_t* pal_map, int lane) {
        // pal_map stores 2-bit palette IDs packed 4 per byte, in ascending dim
        // order.  Lane l owns dims [l*VEC .. l*VEC+VEC-1].  Locate the correct
        // byte and bit offset for this lane's first dim so the formula is valid
        // for any VEC (VEC=2 for HD=64, VEC=4 for HD=128).
        int start_dim  = lane * VEC;
        int byte_idx   = start_dim / 4;       // 4 palette entries per byte
        int bit_shift  = (start_dim % 4) * 2; // 2 bits per entry
        uint8_t my_byte = pal_map[byte_idx];
        uint32_t lane_mask = (1u << lane) - 1;  // bits for lanes < me

        // Part (a): accumulate per-palette cross-lane counts.
        // cross[p] = #{(lane', jj) : lane' < me, palette(lane',jj) == p}
        // i.e. how many dims from earlier lanes map to palette p.
        // Only N_PALETTE popcounts per jj iteration (not VEC), since we
        // accumulate into palette buckets and look up per-j afterwards.
        int cross[N_PALETTE] = {0};
        #pragma unroll
        for (int jj = 0; jj < VEC; jj++) {
            int pjj = (my_byte >> (bit_shift + jj * 2)) & 3;
            uint32_t b0 = __ballot_sync(0xFFFFFFFF, pjj & 1);
            uint32_t b1 = __ballot_sync(0xFFFFFFFF, pjj >> 1);

            #pragma unroll
            for (int p = 0; p < N_PALETTE; p++) {
                uint32_t pal_mask = ((p & 1) ? b0 : ~b0)
                                  & ((p >> 1) ? b1 : ~b1);
                cross[p] += __popc(pal_mask & lane_mask);
            }
        }

        // Part (b) + final scatter: for each of my VEC slots, add
        // lane-local self-contribution (# of earlier own slots with
        // same palette), then combine with cross[p].
        #pragma unroll
        for (int j = 0; j < VEC; j++) {
            int p = (my_byte >> (bit_shift + j * 2)) & 3;
            int local = 0;
            #pragma unroll
            for (int jj = 0; jj < j; jj++) {
                int pjj = (my_byte >> (bit_shift + jj * 2)) & 3;
                local += (pjj == p) ? 1 : 0;
            }
            scatter[j] = (uint8_t)(p * SUB + cross[p] + local);
        }
    }

    __device__ __forceinline__ int operator[](int j) const {
        return scatter[j];
    }
};
