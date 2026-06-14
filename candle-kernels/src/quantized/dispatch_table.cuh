#pragma once

// =============================================================================
// DISPATCH TABLE: Explicit kernel mappings for each batch size
// =============================================================================
// This table defines exactly which kernels to use for each batch size.
// Edit this table to tune dispatch decisions - no procedural logic needed.
//
// Kernel types:
//   S1-S8:        Simple sN kernels (BATCH_TILE=N, single launch with grid.y)
//   S2_ITER2-8:   Iterator kernels (BATCH_TILE=2, NUM_ITERS=N, high occupancy)
//   S3_ITER3:     Iterator kernel (BATCH_TILE=3, NUM_ITERS=3, for multiples of 9)
//   S16_TC:       Tensor core path (BATCH_TILE=16)
//
// Two-kernel entries use kernel1 for bulk, kernel2 for remainder.
// =============================================================================

// Kernel type enum for dispatch table
enum kernel_type_t {
    K_NONE = 0,
    // Simple kernels (use launch_kernel with grid.y tiling)
    K_S1, K_S2, K_S3, K_S4, K_S5, K_S6, K_S7, K_S8,
    // Iterator kernels (use launch_kernel_iter)
    K_S2_ITER2, K_S2_ITER3, K_S2_ITER4,
    K_S2_ITER5, K_S2_ITER6, K_S2_ITER7, K_S2_ITER8,
    K_S3_ITER3,  // Only s3_iter3 is used (for batch=9, 18, 25, 41, 57)
    // TC16 kernels (0-15) - handles batch 1-31 with internal tiling
    // R = batch_size % 16: tc16_0 for pure tc16, tc16_1-15 for tc16+tcR
    K_TC16_0, K_TC16_1, K_TC16_2, K_TC16_3,
    K_TC16_4, K_TC16_5, K_TC16_6, K_TC16_7,
    K_TC16_8, K_TC16_9, K_TC16_10, K_TC16_11,
    K_TC16_12, K_TC16_13, K_TC16_14, K_TC16_15,
    // TC32 kernels (0-15) - handles batch 32+ with greedy internal decomposition
    // R = batch_size % 16: tc32 + optional tc16 + tcR
    K_TC32_0, K_TC32_1, K_TC32_2, K_TC32_3,
    K_TC32_4, K_TC32_5, K_TC32_6, K_TC32_7,
    K_TC32_8, K_TC32_9, K_TC32_10, K_TC32_11,
    K_TC32_12, K_TC32_13, K_TC32_14, K_TC32_15,
};

// Dispatch entry: up to 2 kernels per batch size
struct dispatch_entry_t {
    kernel_type_t k1;      // Primary kernel
    int           b1;      // Batches handled by k1
    kernel_type_t k2;      // Secondary kernel (K_NONE if single launch)
    int           b2;      // Batches handled by k2
};

// =============================================================================
// DISPATCH PLAN: Complete execution plan for a batch size
// =============================================================================
// This structure contains everything needed to execute a matmul dispatch.
// The dispatcher just executes the plan - no decision logic needed.
struct dispatch_plan_t {
    // TC kernel (handles bulk of large batches)
    kernel_type_t tc_kernel;  // K_TC16_*, K_TC32_*, or K_NONE
    int           tc_batch;   // Batches handled by TC
    
    // GEMV kernels (handle remainder after TC, or all batches if no TC)
    kernel_type_t k1;         // Primary GEMV kernel
    int           b1;         // Batches handled by k1
    kernel_type_t k2;         // Secondary GEMV kernel (K_NONE if not needed)
    int           b2;         // Batches handled by k2
    kernel_type_t k3;         // Tertiary GEMV kernel (K_NONE if not needed)
    int           b3;         // Batches handled by k3 (for L2 3-kernel cases)
};

// =============================================================================
// UNIFIED TC DISPATCH PLAN: Enables single kernel launch for multiple TC kernels
// =============================================================================
// Problem: Batch 17 = tc12(12) + tc5(5) currently needs 2 kernel launches,
//          causing L2 cache thrashing (weights read twice → 2x performance cliff)
//
// Solution: Single unified TC kernel launch with a plan that specifies
//           which kernel to run for each batch tile range based on blockIdx.y
//
// Example: Batch 17 → single launch with grid.y=2
//   - blockIdx.y=0: tc12 (batches 0-11, 4 lanes zeroed)
//   - blockIdx.y=1: tc5  (batches 12-16, 11 lanes zeroed)
//
// All TC kernels share the same grid layout: (row_blocks, batch_tiles)
// Each tile = 32 rows × 16 batches (with zero-padding for partial batches)
// =============================================================================

// Maximum number of TC segments in a unified launch
// (2 is sufficient: one tc16 + one tcN remainder, or two tcN kernels)
constexpr int MAX_TC_SEGMENTS = 2;

// Describes one segment of TC work (range of batch tiles)
struct tc_segment_t {
    kernel_type_t kernel;      // K_TC16_*, K_TC32_*, or K_NONE if unused
    int           batch_start; // First batch index this segment handles
    int           batch_count; // Number of batches in this segment (for bounds)
    int           tile_start;  // First blockIdx.y value for this segment
    int           tile_count;  // Number of batch tiles in this segment
};

// Complete unified TC plan
struct tc_unified_plan_t {
    int           total_tiles;                  // Total gridDim.y for the launch
    int           num_segments;                 // Number of active segments (1 or 2)
    tc_segment_t  segments[MAX_TC_SEGMENTS];    // Segment descriptors
};

// =============================================================================
// L2-CACHED PATH: Weights fit in L2 cache
// =============================================================================
// Strategy: Use iter kernels for high occupancy, avoid s1 remainder
// - Even batches: s2_iter for exact fit
// - Multiples of 3: s3_iter or s2_iter+s3
// - Odd non-3: s5, s7, or s2_iter+s3
//
// Table covers batch 0-63. Batch >= 64 uses recursive decomposition.
// =============================================================================
static const dispatch_entry_t L2_DISPATCH_TABLE[64] = {
    // Batch 0-3: Fast path (handled before table lookup)
    /*  0 */ { K_NONE,     0,  K_NONE,  0 },
    /*  1 */ { K_S1,       1,  K_NONE,  0 },
    /*  2 */ { K_S2,       2,  K_NONE,  0 },
    /*  3 */ { K_S3,       3,  K_NONE,  0 },
    
    // Batch 4-7: Simple cases
    /*  4 */ { K_S2_ITER2, 4,  K_NONE,  0 },  // 4 = 2×2
    /*  5 */ { K_S5,       5,  K_NONE,  0 },  // s5 exact
    /*  6 */ { K_S2_ITER3, 6,  K_NONE,  0 },  // 6 = 2×3
    /*  7 */ { K_S7,       7,  K_NONE,  0 },  // s7 exact
    
    // Batch 8-15
    /*  8 */ { K_S2_ITER4, 8,  K_NONE,  0 },  // 8 = 2×4
    /*  9 */ { K_S3_ITER3, 9,  K_NONE,  0 },  // 9 = 3×3
    /* 10 */ { K_S2_ITER5, 10, K_NONE,  0 },  // 10 = 2×5
    /* 11 */ { K_S2_ITER4, 8,  K_S3,    3 },  // 11 = 8+3
    /* 12 */ { K_S2_ITER6, 12, K_NONE,  0 },  // 12 = 2×6
    /* 13 */ { K_S2_ITER5, 10, K_S3,    3 },  // 13 = 10+3
    /* 14 */ { K_S2_ITER7, 14, K_NONE,  0 },  // 14 = 2×7
    /* 15 */ { K_S2_ITER6, 12, K_S3,    3 },  // 15 = 12+3
    
    // Batch 16-23
    /* 16 */ { K_S2_ITER8, 16, K_NONE,  0 },  // 16 = 2×8
    /* 17 */ { K_S2_ITER7, 14, K_S3,    3 },  // 17 = 14+3 (avoid s1)
    /* 18 */ { K_S3_ITER3, 9,  K_S3_ITER3, 9 },  // 18 = 9+9
    /* 19 */ { K_S2_ITER8, 16, K_S3,    3 },  // 19 = 16+3
    /* 20 */ { K_S2_ITER8, 16, K_S2_ITER2, 4 },  // 20 = 16+4
    /* 21 */ { K_S2_ITER8, 16, K_S5,    5 },  // 21 = 16+5
    /* 22 */ { K_S2_ITER8, 16, K_S2_ITER3, 6 },  // 22 = 16+6
    /* 23 */ { K_S2_ITER8, 16, K_S7,    7 },  // 23 = 16+7
    
    // Batch 24-31
    /* 24 */ { K_S2_ITER8, 16, K_S2_ITER4, 8 },  // 24 = 16+8
    /* 25 */ { K_S2_ITER8, 16, K_S3_ITER3, 9 },  // 25 = 16+9
    /* 26 */ { K_S2_ITER8, 16, K_S2_ITER5, 10 }, // 26 = 16+10
    /* 27 */ { K_S2_ITER8, 16, K_S2_ITER4, 8 },  // 27 = 16+8+3 (remainder s3 handled separately)
    /* 28 */ { K_S2_ITER8, 16, K_S2_ITER6, 12 }, // 28 = 16+12
    /* 29 */ { K_S2_ITER8, 16, K_S2_ITER5, 10 }, // 29 = 16+10+3 (remainder s3)
    /* 30 */ { K_S2_ITER8, 16, K_S2_ITER7, 14 }, // 30 = 16+14
    /* 31 */ { K_S2_ITER8, 16, K_S2_ITER6, 12 }, // 31 = 16+12+3 (remainder s3)
    
    // Batch 32-39: s2_iter8 handles 32 with grid.y=2
    /* 32 */ { K_S2_ITER8, 32, K_NONE,  0 },  // 32 = 16×2, grid.y=2
    /* 33 */ { K_S2_ITER8, 30, K_S3,    3 },  // 33 = 30+3 (avoid s1)
    /* 34 */ { K_S2_ITER8, 32, K_S2,    2 },  // 34 = 32+2
    /* 35 */ { K_S2_ITER8, 32, K_S3,    3 },  // 35 = 32+3
    /* 36 */ { K_S2_ITER8, 32, K_S2_ITER2, 4 },  // 36 = 32+4
    /* 37 */ { K_S2_ITER8, 32, K_S5,    5 },  // 37 = 32+5
    /* 38 */ { K_S2_ITER8, 32, K_S2_ITER3, 6 },  // 38 = 32+6
    /* 39 */ { K_S2_ITER8, 32, K_S7,    7 },  // 39 = 32+7
    
    // Batch 40-47
    /* 40 */ { K_S2_ITER8, 32, K_S2_ITER4, 8 },  // 40 = 32+8
    /* 41 */ { K_S2_ITER8, 32, K_S3_ITER3, 9 },  // 41 = 32+9
    /* 42 */ { K_S2_ITER8, 32, K_S2_ITER5, 10 }, // 42 = 32+10
    /* 43 */ { K_S2_ITER8, 32, K_S2_ITER4, 8 },  // 43 = 32+8+3 (remainder s3)
    /* 44 */ { K_S2_ITER8, 32, K_S2_ITER6, 12 }, // 44 = 32+12
    /* 45 */ { K_S2_ITER8, 32, K_S2_ITER5, 10 }, // 45 = 32+10+3 (remainder s3)
    /* 46 */ { K_S2_ITER8, 32, K_S2_ITER7, 14 }, // 46 = 32+14
    /* 47 */ { K_S2_ITER8, 32, K_S2_ITER6, 12 }, // 47 = 32+12+3 (remainder s3)
    
    // Batch 48-55: s2_iter8 handles 48 with grid.y=3
    /* 48 */ { K_S2_ITER8, 48, K_NONE,  0 },  // 48 = 16×3, grid.y=3
    /* 49 */ { K_S2_ITER8, 46, K_S3,    3 },  // 49 = 46+3 (avoid s1)
    /* 50 */ { K_S2_ITER8, 48, K_S2,    2 },  // 50 = 48+2
    /* 51 */ { K_S2_ITER8, 48, K_S3,    3 },  // 51 = 48+3
    /* 52 */ { K_S2_ITER8, 48, K_S2_ITER2, 4 },  // 52 = 48+4
    /* 53 */ { K_S2_ITER8, 48, K_S5,    5 },  // 53 = 48+5
    /* 54 */ { K_S2_ITER8, 48, K_S2_ITER3, 6 },  // 54 = 48+6
    /* 55 */ { K_S2_ITER8, 48, K_S7,    7 },  // 55 = 48+7
    
    // Batch 56-63
    /* 56 */ { K_S2_ITER8, 48, K_S2_ITER4, 8 },  // 56 = 48+8
    /* 57 */ { K_S2_ITER8, 48, K_S3_ITER3, 9 },  // 57 = 48+9
    /* 58 */ { K_S2_ITER8, 48, K_S2_ITER5, 10 }, // 58 = 48+10
    /* 59 */ { K_S2_ITER8, 48, K_S2_ITER4, 8 },  // 59 = 48+8+3 (remainder s3)
    /* 60 */ { K_S2_ITER8, 48, K_S2_ITER6, 12 }, // 60 = 48+12
    /* 61 */ { K_S2_ITER8, 48, K_S2_ITER5, 10 }, // 61 = 48+10+3 (remainder s3)
    /* 62 */ { K_S2_ITER8, 48, K_S2_ITER7, 14 }, // 62 = 48+14
    /* 63 */ { K_S2_ITER8, 48, K_S2_ITER6, 12 }, // 63 = 48+12+3 (remainder s3)
};

// Remainder for entries that need 3 kernels (b1+b2 < batch)
// Returns the remainder that needs a final s3 launch, or 0 if none
inline int get_l2_remainder(int batch) {
    // Pattern: 27,29,31, 43,45,47, 59,61,63 all need +3
    int mod16 = batch % 16;
    if (batch >= 27 && (mod16 == 11 || mod16 == 13 || mod16 == 15)) {
        return 3;
    }
    return 0;
}

// =============================================================================
// DRAM-BOUND PATH: Weights exceed L2 cache
// =============================================================================
// Strategy: Use simple sN kernels with grid.y tiling for weight reuse
// - Prefer exact divisibility (single launch)
// - Fall back to sN + sM decomposition avoiding s1 where possible
//
// Table covers batch 0-63. Batch >= 64 uses recursive decomposition.
// =============================================================================
static const dispatch_entry_t DRAM_DISPATCH_TABLE[64] = {
    // Batch 0-3: Fast path (handled before table lookup)
    /*  0 */ { K_NONE,  0,  K_NONE, 0 },
    /*  1 */ { K_S1,    1,  K_NONE, 0 },
    /*  2 */ { K_S2,    2,  K_NONE, 0 },
    /*  3 */ { K_S3,    3,  K_NONE, 0 },
    
    // Batch 4-7: Exact fit
    /*  4 */ { K_S4,    4,  K_NONE, 0 },
    /*  5 */ { K_S5,    5,  K_NONE, 0 },
    /*  6 */ { K_S6,    6,  K_NONE, 0 },
    /*  7 */ { K_S7,    7,  K_NONE, 0 },
    
    // Batch 8-15
    /*  8 */ { K_S8,    8,  K_NONE, 0 },
    /*  9 */ { K_S3,    9,  K_NONE, 0 },  // 9%3=0, grid.y=3
    /* 10 */ { K_S5,   10,  K_NONE, 0 },  // 10%5=0, grid.y=2
    /* 11 */ { K_S8,    8,  K_S3,   3 },  // 11 = 8+3
    /* 12 */ { K_S6,   12,  K_NONE, 0 },  // 12%6=0, grid.y=2
    /* 13 */ { K_S8,    8,  K_S5,   5 },  // 13 = 8+5
    /* 14 */ { K_S7,   14,  K_NONE, 0 },  // 14%7=0, grid.y=2
    /* 15 */ { K_S5,   15,  K_NONE, 0 },  // 15%5=0, grid.y=3
    
    // Batch 16-23
    /* 16 */ { K_S8,   16,  K_NONE, 0 },  // 16%8=0, grid.y=2
    /* 17 */ { K_S7,   14,  K_S3,   3 },  // 17 = 14+3 (avoid s1)
    /* 18 */ { K_S6,   18,  K_NONE, 0 },  // 18%6=0, grid.y=3
    /* 19 */ { K_S8,   16,  K_S3,   3 },  // 19 = 16+3
    /* 20 */ { K_S5,   20,  K_NONE, 0 },  // 20%5=0, grid.y=4
    /* 21 */ { K_S7,   21,  K_NONE, 0 },  // 21%7=0, grid.y=3
    /* 22 */ { K_S8,   16,  K_S6,   6 },  // 22 = 16+6
    /* 23 */ { K_S8,   16,  K_S7,   7 },  // 23 = 16+7
    
    // Batch 24-31
    /* 24 */ { K_S8,   24,  K_NONE, 0 },  // 24%8=0, grid.y=3
    /* 25 */ { K_S5,   25,  K_NONE, 0 },  // 25%5=0, grid.y=5
    /* 26 */ { K_S8,   24,  K_S2,   2 },  // 26 = 24+2
    /* 27 */ { K_S3,   27,  K_NONE, 0 },  // 27%3=0, grid.y=9
    /* 28 */ { K_S7,   28,  K_NONE, 0 },  // 28%7=0, grid.y=4
    /* 29 */ { K_S8,   24,  K_S5,   5 },  // 29 = 24+5
    /* 30 */ { K_S6,   30,  K_NONE, 0 },  // 30%6=0, grid.y=5
    /* 31 */ { K_S8,   24,  K_S7,   7 },  // 31 = 24+7
    
    // Batch 32-39
    /* 32 */ { K_S8,   32,  K_NONE, 0 },  // 32%8=0, grid.y=4
    /* 33 */ { K_S3,   33,  K_NONE, 0 },  // 33%3=0, grid.y=11
    /* 34 */ { K_S8,   32,  K_S2,   2 },  // 34 = 32+2
    /* 35 */ { K_S7,   35,  K_NONE, 0 },  // 35%7=0, grid.y=5
    /* 36 */ { K_S6,   36,  K_NONE, 0 },  // 36%6=0, grid.y=6
    /* 37 */ { K_S8,   32,  K_S5,   5 },  // 37 = 32+5
    /* 38 */ { K_S8,   32,  K_S6,   6 },  // 38 = 32+6
    /* 39 */ { K_S3,   39,  K_NONE, 0 },  // 39%3=0, grid.y=13
    
    // Batch 40-47
    /* 40 */ { K_S8,   40,  K_NONE, 0 },  // 40%8=0, grid.y=5
    /* 41 */ { K_S7,   35,  K_S6,   6 },  // 41 = 35+6 (avoid s1)
    /* 42 */ { K_S7,   42,  K_NONE, 0 },  // 42%7=0, grid.y=6
    /* 43 */ { K_S8,   40,  K_S3,   3 },  // 43 = 40+3
    /* 44 */ { K_S8,   40,  K_S4,   4 },  // 44 = 40+4
    /* 45 */ { K_S5,   45,  K_NONE, 0 },  // 45%5=0, grid.y=9
    /* 46 */ { K_S8,   40,  K_S6,   6 },  // 46 = 40+6
    /* 47 */ { K_S8,   40,  K_S7,   7 },  // 47 = 40+7
    
    // Batch 48-55
    /* 48 */ { K_S8,   48,  K_NONE, 0 },  // 48%8=0, grid.y=6
    /* 49 */ { K_S7,   49,  K_NONE, 0 },  // 49%7=0, grid.y=7
    /* 50 */ { K_S5,   50,  K_NONE, 0 },  // 50%5=0, grid.y=10
    /* 51 */ { K_S3,   51,  K_NONE, 0 },  // 51%3=0, grid.y=17
    /* 52 */ { K_S8,   48,  K_S4,   4 },  // 52 = 48+4
    /* 53 */ { K_S8,   48,  K_S5,   5 },  // 53 = 48+5
    /* 54 */ { K_S6,   54,  K_NONE, 0 },  // 54%6=0, grid.y=9
    /* 55 */ { K_S8,   48,  K_S7,   7 },  // 55 = 48+7
    
    // Batch 56-63
    /* 56 */ { K_S8,   56,  K_NONE, 0 },  // 56%8=0, grid.y=7
    /* 57 */ { K_S3,   57,  K_NONE, 0 },  // 57%3=0, grid.y=19
    /* 58 */ { K_S8,   56,  K_S2,   2 },  // 58 = 56+2
    /* 59 */ { K_S8,   56,  K_S3,   3 },  // 59 = 56+3
    /* 60 */ { K_S6,   60,  K_NONE, 0 },  // 60%6=0, grid.y=10
    /* 61 */ { K_S8,   56,  K_S5,   5 },  // 61 = 56+5
    /* 62 */ { K_S8,   56,  K_S6,   6 },  // 62 = 56+6
    /* 63 */ { K_S7,   63,  K_NONE, 0 },  // 63%7=0, grid.y=9
};

// =============================================================================
// HELPER FUNCTIONS (don't depend on kernel_set_t)
// =============================================================================

// Is this an iterator kernel? (needs launch_kernel_iter)
inline bool is_iter_kernel(kernel_type_t kt) {
    return (kt >= K_S2_ITER2 && kt <= K_S3_ITER3);
}

// Get batches per block for iterator kernels
inline int get_iter_batches_per_block(kernel_type_t kt) {
    switch (kt) {
        case K_S2_ITER2: return 4;
        case K_S2_ITER3: return 6;
        case K_S2_ITER4: return 8;
        case K_S2_ITER5: return 10;
        case K_S2_ITER6: return 12;
        case K_S2_ITER7: return 14;
        case K_S2_ITER8: return 16;
        case K_S3_ITER3: return 9;
        default: return 0;
    }
}

// Get batch tile for simple kernels
inline int get_batch_tile(kernel_type_t kt) {
    switch (kt) {
        case K_S1: return 1;
        case K_S2: return 2;
        case K_S3: return 3;
        case K_S4: return 4;
        case K_S5: return 5;
        case K_S6: return 6;
        case K_S7: return 7;
        case K_S8: return 8;
        default: return 0;
    }
}

// Check if kernel type is a TC16 kernel (0-15)
inline bool is_tc16_kernel(kernel_type_t kt) {
    return kt >= K_TC16_0 && kt <= K_TC16_15;
}

// Check if kernel type is a TC32 kernel (0-15)
inline bool is_tc32_kernel(kernel_type_t kt) {
    return kt >= K_TC32_0 && kt <= K_TC32_15;
}

// Get remainder number for TC16 dispatch (0-15)
// Returns -1 for non-TC16 kernels
inline int get_tc16_remainder(kernel_type_t kt) {
    if (kt >= K_TC16_0 && kt <= K_TC16_15) {
        return kt - K_TC16_0;
    }
    return -1;
}

// Get remainder number for TC32 dispatch (0-15)
// Returns -1 for non-TC32 kernels
inline int get_tc32_remainder(kernel_type_t kt) {
    if (kt >= K_TC32_0 && kt <= K_TC32_15) {
        return kt - K_TC32_0;
    }
    return -1;
}

// =============================================================================
// TC PATH: Tensor core dispatch for batch 1-31 (uses tc16 kernels)
// =============================================================================
// Strategy: tc16 for ALL batch 1-31. Benchmarking found the TC kernels faster
// than the CUDA-core GEMV (s1/s2) path in every case, batch 1-2 included, so
// get_dispatch_plan() routes every batch >= 1 to tc16. This static table is the
// legacy pre-SM80 mapping and is no longer consulted on the TC path;
// get_dispatch_plan() is the source of truth. The batch 1-2 K_S1/K_S2 entries
// below apply only to GPUs without tensor cores.
//
// - Batch 3-15: tc16_N where N = batch_size (single kernel)
// - Batch 16-31: tc16_N where N = batch_size % 16 (single kernel, internal tiling)
//
// The tc16 kernels compute tc16_tiles and remainder internally from batch_size.
// =============================================================================
static const dispatch_entry_t TC_DISPATCH_TABLE[32] = {
    // Batch 0: nothing to do
    /*  0 */ { K_NONE,      0,  K_NONE,  0 },

    // Batch 1-2: GEMV only on pre-SM80 GPUs; TC hardware uses tc16_1/tc16_2.
    /*  1 */ { K_S1,        1,  K_NONE,  0 },
    /*  2 */ { K_S2,        2,  K_NONE,  0 },
    
    // Batch 3-15: Single tc16 kernel (no tc16 tiles, just remainder)
    /*  3 */ { K_TC16_3,    3,  K_NONE,  0 },
    /*  4 */ { K_TC16_4,    4,  K_NONE,  0 },
    /*  5 */ { K_TC16_5,    5,  K_NONE,  0 },
    /*  6 */ { K_TC16_6,    6,  K_NONE,  0 },
    /*  7 */ { K_TC16_7,    7,  K_NONE,  0 },
    /*  8 */ { K_TC16_8,    8,  K_NONE,  0 },
    /*  9 */ { K_TC16_9,    9,  K_NONE,  0 },
    /* 10 */ { K_TC16_10,  10,  K_NONE,  0 },
    /* 11 */ { K_TC16_11,  11,  K_NONE,  0 },
    /* 12 */ { K_TC16_12,  12,  K_NONE,  0 },
    /* 13 */ { K_TC16_13,  13,  K_NONE,  0 },
    /* 14 */ { K_TC16_14,  14,  K_NONE,  0 },
    /* 15 */ { K_TC16_15,  15,  K_NONE,  0 },
    
    // Batch 16-31: tc16 with internal tc16 + tcR dispatch
    /* 16 */ { K_TC16_0,   16,  K_NONE,  0 },  // tc16 only (R=0)
    /* 17 */ { K_TC16_1,   17,  K_NONE,  0 },  // tc16 + tc1
    /* 18 */ { K_TC16_2,   18,  K_NONE,  0 },  // tc16 + tc2
    /* 19 */ { K_TC16_3,   19,  K_NONE,  0 },  // tc16 + tc3
    /* 20 */ { K_TC16_4,   20,  K_NONE,  0 },  // tc16 + tc4
    /* 21 */ { K_TC16_5,   21,  K_NONE,  0 },  // tc16 + tc5
    /* 22 */ { K_TC16_6,   22,  K_NONE,  0 },  // tc16 + tc6
    /* 23 */ { K_TC16_7,   23,  K_NONE,  0 },  // tc16 + tc7
    /* 24 */ { K_TC16_8,   24,  K_NONE,  0 },  // tc16 + tc8
    /* 25 */ { K_TC16_9,   25,  K_NONE,  0 },  // tc16 + tc9
    /* 26 */ { K_TC16_10,  26,  K_NONE,  0 },  // tc16 + tc10
    /* 27 */ { K_TC16_11,  27,  K_NONE,  0 },  // tc16 + tc11
    /* 28 */ { K_TC16_12,  28,  K_NONE,  0 },  // tc16 + tc12
    /* 29 */ { K_TC16_13,  29,  K_NONE,  0 },  // tc16 + tc13
    /* 30 */ { K_TC16_14,  30,  K_NONE,  0 },  // tc16 + tc14
    /* 31 */ { K_TC16_15,  31,  K_NONE,  0 },  // tc16 + tc15
};

// =============================================================================
// TC32 PATH: Tensor core dispatch for batch 32+ (using tc32 kernels)
// =============================================================================
// Strategy: Use tc32 which computes greedy decomposition internally:
// - tc32 tiles handle multiples of 32
// - tc16 tile handles remainder >= 16
// - tcR handles final remainder 0-15
//
// Kernel selection: R = batch_size % 16
// Grid.y computed internally based on batch_size
// =============================================================================
static const dispatch_entry_t TC32_DISPATCH_TABLE[64] = {
    // Batch 0-31: Handled by TC_DISPATCH_TABLE (uses tc16 kernels)
    /*  0 */ { K_NONE,       0,  K_NONE,  0 },
    /*  1 */ { K_S1,         1,  K_NONE,  0 },
    /*  2 */ { K_S2,         2,  K_NONE,  0 },
    /*  3 */ { K_TC16_3,     3,  K_NONE,  0 },
    /*  4 */ { K_TC16_4,     4,  K_NONE,  0 },
    /*  5 */ { K_TC16_5,     5,  K_NONE,  0 },
    /*  6 */ { K_TC16_6,     6,  K_NONE,  0 },
    /*  7 */ { K_TC16_7,     7,  K_NONE,  0 },
    /*  8 */ { K_TC16_8,     8,  K_NONE,  0 },
    /*  9 */ { K_TC16_9,     9,  K_NONE,  0 },
    /* 10 */ { K_TC16_10,   10,  K_NONE,  0 },
    /* 11 */ { K_TC16_11,   11,  K_NONE,  0 },
    /* 12 */ { K_TC16_12,   12,  K_NONE,  0 },
    /* 13 */ { K_TC16_13,   13,  K_NONE,  0 },
    /* 14 */ { K_TC16_14,   14,  K_NONE,  0 },
    /* 15 */ { K_TC16_15,   15,  K_NONE,  0 },
    /* 16 */ { K_TC16_0,    16,  K_NONE,  0 },
    /* 17 */ { K_TC16_1,    17,  K_NONE,  0 },
    /* 18 */ { K_TC16_2,    18,  K_NONE,  0 },
    /* 19 */ { K_TC16_3,    19,  K_NONE,  0 },
    /* 20 */ { K_TC16_4,    20,  K_NONE,  0 },
    /* 21 */ { K_TC16_5,    21,  K_NONE,  0 },
    /* 22 */ { K_TC16_6,    22,  K_NONE,  0 },
    /* 23 */ { K_TC16_7,    23,  K_NONE,  0 },
    /* 24 */ { K_TC16_8,    24,  K_NONE,  0 },
    /* 25 */ { K_TC16_9,    25,  K_NONE,  0 },
    /* 26 */ { K_TC16_10,   26,  K_NONE,  0 },
    /* 27 */ { K_TC16_11,   27,  K_NONE,  0 },
    /* 28 */ { K_TC16_12,   28,  K_NONE,  0 },
    /* 29 */ { K_TC16_13,   29,  K_NONE,  0 },
    /* 30 */ { K_TC16_14,   30,  K_NONE,  0 },
    /* 31 */ { K_TC16_15,   31,  K_NONE,  0 },
    
    // Batch 32-63: tc32 kernels with greedy internal decomposition
    // R = batch % 16, kernel computes tc32_tiles, has_tc16, remainder internally
    /* 32 */ { K_TC32_0,    32,  K_NONE,  0 },  // tc32(32) only
    /* 33 */ { K_TC32_1,    33,  K_NONE,  0 },  // tc32(32) + tc1(1)
    /* 34 */ { K_TC32_2,    34,  K_NONE,  0 },  // tc32(32) + tc2(2)
    /* 35 */ { K_TC32_3,    35,  K_NONE,  0 },  // tc32(32) + tc3(3)
    /* 36 */ { K_TC32_4,    36,  K_NONE,  0 },  // tc32(32) + tc4(4)
    /* 37 */ { K_TC32_5,    37,  K_NONE,  0 },  // tc32(32) + tc5(5)
    /* 38 */ { K_TC32_6,    38,  K_NONE,  0 },  // tc32(32) + tc6(6)
    /* 39 */ { K_TC32_7,    39,  K_NONE,  0 },  // tc32(32) + tc7(7)
    /* 40 */ { K_TC32_8,    40,  K_NONE,  0 },  // tc32(32) + tc8(8)
    /* 41 */ { K_TC32_9,    41,  K_NONE,  0 },  // tc32(32) + tc9(9)
    /* 42 */ { K_TC32_10,   42,  K_NONE,  0 },  // tc32(32) + tc10(10)
    /* 43 */ { K_TC32_11,   43,  K_NONE,  0 },  // tc32(32) + tc11(11)
    /* 44 */ { K_TC32_12,   44,  K_NONE,  0 },  // tc32(32) + tc12(12)
    /* 45 */ { K_TC32_13,   45,  K_NONE,  0 },  // tc32(32) + tc13(13)
    /* 46 */ { K_TC32_14,   46,  K_NONE,  0 },  // tc32(32) + tc14(14)
    /* 47 */ { K_TC32_15,   47,  K_NONE,  0 },  // tc32(32) + tc15(15)
    /* 48 */ { K_TC32_0,    48,  K_NONE,  0 },  // tc32(32) + tc16(16)
    /* 49 */ { K_TC32_1,    49,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc1(1)
    /* 50 */ { K_TC32_2,    50,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc2(2)
    /* 51 */ { K_TC32_3,    51,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc3(3)
    /* 52 */ { K_TC32_4,    52,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc4(4)
    /* 53 */ { K_TC32_5,    53,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc5(5)
    /* 54 */ { K_TC32_6,    54,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc6(6)
    /* 55 */ { K_TC32_7,    55,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc7(7)
    /* 56 */ { K_TC32_8,    56,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc8(8)
    /* 57 */ { K_TC32_9,    57,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc9(9)
    /* 58 */ { K_TC32_10,   58,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc10(10)
    /* 59 */ { K_TC32_11,   59,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc11(11)
    /* 60 */ { K_TC32_12,   60,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc12(12)
    /* 61 */ { K_TC32_13,   61,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc13(13)
    /* 62 */ { K_TC32_14,   62,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc14(14)
    /* 63 */ { K_TC32_15,   63,  K_NONE,  0 },  // tc32(32) + tc16(16) + tc15(15)
};

// =============================================================================
// LOOKUP HELPERS
// =============================================================================

// Table size constants
constexpr int DISPATCH_TABLE_SIZE = 64;
constexpr int TC_TABLE_SIZE = 32;
constexpr int TC32_TABLE_SIZE = 64;

// Thresholds for tensor core kernels
constexpr int TC16_MIN_BATCH = 16;     // TC16 requires batch >= 16
constexpr int TC_MIN_BATCH = 1;        // TC kernels available for batch >= 1

// =============================================================================
// UNIFIED DISPATCH PLAN
// =============================================================================

// Get complete dispatch plan for a batch size
// Simplified for tc16/tc32 kernels - single kernel dispatch for TC path
inline dispatch_plan_t get_dispatch_plan(int batch_size, bool use_l2_path, bool use_tc) {
    dispatch_plan_t plan = { K_NONE, 0, K_NONE, 0, K_NONE, 0, K_NONE, 0 };
    
    // TC path: use tc16/tc32 kernels which compute tiling internally.
    // ALL batch >= 1 goes through tensor cores, including batch 1-2: benchmarking
    // found tc16_1/tc16_2 faster than the CUDA-core s1/s2 GEMV in every case.
    if (use_tc && batch_size >= 1) {
        if (batch_size >= 32) {
            // Use tc32 - kernel computes tc32 + tc16 + tcR internally
            int remainder = batch_size % 16;
            plan.tc_kernel = (kernel_type_t)(K_TC32_0 + remainder);
            plan.tc_batch = batch_size;
            // No GEMV remainder - tc32 kernel handles everything
            return plan;
        } else {
            // Use tc16 - kernel computes tc16 + tcR internally
            int remainder = batch_size % 16;
            if (batch_size >= 16) {
                // batch 16-31: uses tc16 + tcR
                plan.tc_kernel = (kernel_type_t)(K_TC16_0 + remainder);
            } else {
                // batch 1-15: uses tcR only (no tc16 tiles)
                plan.tc_kernel = (kernel_type_t)(K_TC16_0 + batch_size);
            }
            plan.tc_batch = batch_size;
            return plan;
        }
    }
    
    // Non-TC path or batch < 3: use GEMV tables
    if (batch_size <= 0) {
        return plan;
    }
    
    if (batch_size < DISPATCH_TABLE_SIZE) {
        const dispatch_entry_t& entry = use_l2_path ? 
            L2_DISPATCH_TABLE[batch_size] : DRAM_DISPATCH_TABLE[batch_size];
        plan.k1 = entry.k1;
        plan.b1 = entry.b1;
        plan.k2 = entry.k2;
        plan.b2 = entry.b2;
    } else {
        // Recursive decomposition for large batches
        plan.k1 = K_S8;
        plan.b1 = 64;
    }
    
    return plan;
}

// Get dispatch entry for a given batch size and path
inline const dispatch_entry_t& get_dispatch_entry(int batch, bool use_l2_path) {
    if (batch >= DISPATCH_TABLE_SIZE) {
        static const dispatch_entry_t fallback = { K_NONE, 0, K_NONE, 0 };
        return fallback;
    }
    return use_l2_path ? L2_DISPATCH_TABLE[batch] : DRAM_DISPATCH_TABLE[batch];
}

// Get kernel name as string for reporting
inline const char* kernel_type_name(kernel_type_t kt) {
    switch (kt) {
        case K_NONE: return "";
        case K_S1: return "s1";
        case K_S2: return "s2";
        case K_S3: return "s3";
        case K_S4: return "s4";
        case K_S5: return "s5";
        case K_S6: return "s6";
        case K_S7: return "s7";
        case K_S8: return "s8";
        case K_S2_ITER2: return "s2i2";
        case K_S2_ITER3: return "s2i3";
        case K_S2_ITER4: return "s2i4";
        case K_S2_ITER5: return "s2i5";
        case K_S2_ITER6: return "s2i6";
        case K_S2_ITER7: return "s2i7";
        case K_S2_ITER8: return "s2i8";
        case K_S3_ITER3: return "s3i3";
        case K_TC16_0: return "tc16_0";
        case K_TC16_1: return "tc16_1";
        case K_TC16_2: return "tc16_2";
        case K_TC16_3: return "tc16_3";
        case K_TC16_4: return "tc16_4";
        case K_TC16_5: return "tc16_5";
        case K_TC16_6: return "tc16_6";
        case K_TC16_7: return "tc16_7";
        case K_TC16_8: return "tc16_8";
        case K_TC16_9: return "tc16_9";
        case K_TC16_10: return "tc16_10";
        case K_TC16_11: return "tc16_11";
        case K_TC16_12: return "tc16_12";
        case K_TC16_13: return "tc16_13";
        case K_TC16_14: return "tc16_14";
        case K_TC16_15: return "tc16_15";
        case K_TC32_0: return "tc32_0";
        case K_TC32_1: return "tc32_1";
        case K_TC32_2: return "tc32_2";
        case K_TC32_3: return "tc32_3";
        case K_TC32_4: return "tc32_4";
        case K_TC32_5: return "tc32_5";
        case K_TC32_6: return "tc32_6";
        case K_TC32_7: return "tc32_7";
        case K_TC32_8: return "tc32_8";
        case K_TC32_9: return "tc32_9";
        case K_TC32_10: return "tc32_10";
        case K_TC32_11: return "tc32_11";
        case K_TC32_12: return "tc32_12";
        case K_TC32_13: return "tc32_13";
        case K_TC32_14: return "tc32_14";
        case K_TC32_15: return "tc32_15";
        default: return "?";
    }
}
