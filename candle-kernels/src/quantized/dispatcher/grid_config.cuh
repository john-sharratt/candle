// The hierarchical grid configuration that keeps each weight's tiles
// cache-resident.
//
// `compute_grid_config` builds the grid under the L2 capacity constraint.
// `GRID_LAYOUT_ROW_FAST` and `GRID_LAYOUT_BATCH_FAST` select which dimension
// varies fastest, and `compute_grid` derives the grid dimensions for a layout.

static int g_cached_blocks_per_sm = 8;      // Max concurrent blocks per SM (occupancy)

// =============================================================================
// GRID-AS-CACHE-HIERARCHY CONFIGURATION
// =============================================================================
// Implements hierarchical L1/L2 cache optimization for TC kernels:
//   gridDim.x = batch tiles (L1 scope): Same-SM blocks share weights in L1
//   gridDim.y = row tiles (L2 scope): All blocks share activations in L2
//   gridDim.z = wave overflow (sequential processing)
//
// L2 capacity constraint ensures activation working set fits in L2:
//   act_per_x_group = gridDim.x × TILE_BATCH × K × sizeof(half)
//   y_from_l2 = (L2_size × 0.70) / act_per_x_group
//
// Z-stride encoding (rows inner for L2 activation persistence):
//   z = row_group + batch_group × num_row_groups

struct GridConfig {
    dim3 grid;          // Final grid dimensions (x, y, z)
    int row_groups;     // Number of row groups for z-decode
    int batch_groups;   // Number of batch groups for z-decode
    int wave_size;      // Blocks per wave (SM count × blocks/SM)
};

// Compute hierarchical grid configuration with L2 capacity constraint
// Parameters:
//   row_tiles: Total row tiles to process
//   batch_tiles: Total batch tiles to process
//   tile_batch: Batch size per tile (16 or 32)
//   K: Hidden dimension (ncols_x)
__host__ inline GridConfig compute_grid_config(
    int row_tiles, int batch_tiles, int tile_batch, int K
) {
    GridConfig cfg;
    const auto& caps = get_device_caps();
    cfg.wave_size = caps.sm_count * g_cached_blocks_per_sm;

    // L2-usable size (90% of L2)
    const size_t l2_usable = (caps.l2_cache_size * 90) / 100;
    const size_t tile_size = (size_t)tile_batch * K * sizeof(half);
    // Total number of tiles that fit in L2
    const double A = (tile_size > 0) ? (double)l2_usable / (double)tile_size : 1.0;

    // Square-root-based sizing for grid.x and grid.y
    int grid_x = (int)ceil(sqrt(A));
    grid_x = max(1, min(grid_x, batch_tiles));
    // Cap grid.x by occupancy
    grid_x = min(grid_x, cfg.wave_size);

    int grid_y = (int)max(1.0, floor(A / grid_x));
    grid_y = min(grid_y, row_tiles);
    // Cap grid.y by occupancy
    if (grid_x > 0) grid_y = min(grid_y, cfg.wave_size / grid_x);

    // Row and batch groups for z-dimension
    cfg.row_groups = (row_tiles + grid_y - 1) / grid_y;
    cfg.batch_groups = (batch_tiles + grid_x - 1) / grid_x;

    // grid.z: Total waves (rows inner, batches outer for L2 persistence)
    const int grid_z = cfg.row_groups * cfg.batch_groups;

    cfg.grid = dim3(grid_x, grid_y, grid_z);
    return cfg;
}

// Grid layout modes for L2 cache optimization:
//   GRID_LAYOUT_ROW_FAST (0): blockIdx.x=row, blockIdx.y=batch
//     - Y (activations) stay in L2 across row blocks
//     - Best when batch_tiles=1 (most decode scenarios)
//   GRID_LAYOUT_BATCH_FAST (1): blockIdx.x=batch, blockIdx.y=row  
//     - X (weights) stay in L2 across batch tiles
//     - Best when batch_tiles>1 and weights don't fit in L2
constexpr int GRID_LAYOUT_ROW_FAST = 0;
constexpr int GRID_LAYOUT_BATCH_FAST = 1;

// Helper to compute grid dimensions based on layout
__host__ inline dim3 compute_grid(int nrows_x, int batch_size, int batch_tile, int grid_layout) {
    int row_blocks = (nrows_x + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK;
    int batch_tiles = (batch_size + batch_tile - 1) / batch_tile;
    if (grid_layout == GRID_LAYOUT_BATCH_FAST) {
        return dim3(batch_tiles, row_blocks);  // batch varies fastest
    } else {
        return dim3(row_blocks, batch_tiles);  // row varies fastest (default)
    }
}
