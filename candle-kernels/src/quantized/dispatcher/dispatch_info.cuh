// The dispatch-info query that benchmark reporting uses.
//
// `get_dispatch_info` answers the dispatch-info query used for benchmark
// reporting.

// =============================================================================
// DISPATCH INFO QUERY (for benchmark reporting)
// =============================================================================
// Returns a string describing which kernels will be used for a given configuration.
// Format: "kernel1(batch1)+kernel2(batch2)" or "kernel1(batch1)" for single launches
// Examples: "s2i8(16)", "s2i4(8)+s3(3)", "[tc32(32)]+s8(8)"
//
// This function uses get_dispatch_plan() to ensure reporting matches actual dispatch.

extern "C" int32_t get_dispatch_info(
    int32_t batch_size,
    size_t weight_bytes,
    char* buffer,
    int32_t buffer_len
) {
    if (buffer == nullptr || buffer_len < 32) {
        return -1;
    }
    
    // Query L2 cache size and SM version
    int device;
    cudaGetDevice(&device);
    int l2_cache_size;
    cudaDeviceGetAttribute(&l2_cache_size, cudaDevAttrL2CacheSize, device);
    int sm_major;
    cudaDeviceGetAttribute(&sm_major, cudaDevAttrComputeCapabilityMajor, device);
    
    // Use 80% of L2 as threshold (same as dispatcher)
    size_t l2_threshold = (size_t)(l2_cache_size * 0.8);
    bool use_l2_path = (weight_bytes <= l2_threshold);
    bool use_tc = (sm_major >= 8);  // Ampere+ for tensor cores
    
    int pos = 0;
    int remaining = batch_size;
    
    // =========================================================================
    // PLAN-BASED REPORTING: Uses same logic as dispatcher
    // =========================================================================
    
    // For batch >= 64, we need bulk reduction first
    // L2 path: s2_iter8 (16 batches/call), DRAM path: s8 (8 batches/call)
    if (remaining >= DISPATCH_TABLE_SIZE) {
        // Get plan for the full batch - TC handles up front
        dispatch_plan_t plan = get_dispatch_plan(remaining, use_l2_path, use_tc);
        
        // Report TC kernel if used - TC32 handles everything internally
        if (plan.tc_kernel != K_NONE && plan.tc_batch > 0) {
            pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                           kernel_type_name(plan.tc_kernel), plan.tc_batch);
            remaining = 0;  // TC32 kernel handles all batches internally
        }
        
        // Report bulk reduction for remaining >= 64 (only if no TC)
        if (remaining >= DISPATCH_TABLE_SIZE) {
            if (use_l2_path) {
                int bulk = (remaining / 16) * 16;
                pos += snprintf(buffer + pos, buffer_len - pos, "s2i8(%d)", bulk);
                remaining -= bulk;
            } else {
                int bulk = (remaining / 8) * 8;
                pos += snprintf(buffer + pos, buffer_len - pos, "s8(%d)", bulk);
                remaining -= bulk;
            }
            if (remaining > 0 && pos < buffer_len - 1) {
                buffer[pos++] = '+';
            }
        }
    }
    
    // =========================================================================
    // TABLE LOOKUP for remainder (0-63)
    // =========================================================================
    if (remaining > 0 && remaining < DISPATCH_TABLE_SIZE) {
        dispatch_plan_t plan = get_dispatch_plan(remaining, use_l2_path, use_tc);
        
        // TC kernel handles everything internally (tc16 tiles + remainder)
        if (plan.tc_kernel != K_NONE && plan.tc_batch > 0) {
            pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                           kernel_type_name(plan.tc_kernel), plan.tc_batch);
            remaining = 0;  // TC kernel handles all batches, no GEMV remainder
        } else {
            // GEMV kernels (only when TC not used)
            // First GEMV kernel
            if (plan.k1 != K_NONE && plan.b1 > 0) {
                pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                               kernel_type_name(plan.k1), plan.b1);
                remaining -= plan.b1;
            }
            
            // Second GEMV kernel
            if (plan.k2 != K_NONE && plan.b2 > 0 && pos < buffer_len - 1) {
                buffer[pos++] = '+';
                pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                               kernel_type_name(plan.k2), plan.b2);
                remaining -= plan.b2;
            }
            
            // Third GEMV kernel (L2 path 3-kernel cases)
            if (plan.k3 != K_NONE && plan.b3 > 0 && pos < buffer_len - 1) {
                buffer[pos++] = '+';
                pos += snprintf(buffer + pos, buffer_len - pos, "%s(%d)", 
                               kernel_type_name(plan.k3), plan.b3);
                remaining -= plan.b3;
            }
            
            // Check for 3-kernel case (L2 path remainder) - fallback if not in plan
            if (use_l2_path && remaining > 0) {
                int l2_rem = get_l2_remainder(batch_size % 64);
                if (l2_rem > 0 && remaining == l2_rem && pos < buffer_len - 1) {
                    buffer[pos++] = '+';
                    pos += snprintf(buffer + pos, buffer_len - pos, "s3(%d)", l2_rem);
                    remaining = 0;
                }
            }
        }
    }
    
    // Final remainder (shouldn't happen with good tables)
    if (remaining > 0 && remaining <= 7 && pos < buffer_len - 8) {
        if (pos > 0 && buffer[pos-1] != '+') {
            buffer[pos++] = '+';
        }
        pos += snprintf(buffer + pos, buffer_len - pos, "s%d(%d)", remaining, remaining);
    }
    
    buffer[pos] = '\0';
    return pos;
}
