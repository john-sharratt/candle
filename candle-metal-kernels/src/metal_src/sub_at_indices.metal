#include <metal_stdlib>

using namespace metal;

// Sub at indices - parallelizes over indices for sparse updates
kernel void sub_at_indices_f32(
    constant size_t &num_elements,
    constant size_t &vocab_size,
    constant uint32_t *indices,
    constant size_t &num_indices,
    constant float &value,
    device const float *input,
    device float *output,
    uint tid [[ thread_position_in_grid ]],
    uint tsize [[ threads_per_grid ]]
) {
    // Copy input to output
    for (uint id = tid; id < num_elements; id += tsize) {
        output[id] = input[id];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    // Parallel update over indices
    if (tid < num_indices) {
        uint32_t idx = indices[tid];
        
        size_t batch_size = num_elements / vocab_size;
        
        for (size_t batch = 0; batch < batch_size; batch++) {
            size_t offset = batch * vocab_size + idx;
            atomic_fetch_add_explicit((device atomic<float>*)&output[offset], -value, memory_order_relaxed);
        }
    }
}

kernel void sub_at_indices_f16(
    constant size_t &num_elements,
    constant size_t &vocab_size,
    constant uint32_t *indices,
    constant size_t &num_indices,
    constant float &value,
    device const half *input,
    device half *output,
    uint tid [[ thread_position_in_grid ]],
    uint tsize [[ threads_per_grid ]]
) {
    for (uint id = tid; id < num_elements; id += tsize) {
        output[id] = input[id];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    if (tid < num_indices) {
        uint32_t idx = indices[tid];
        half value_h = half(value);
        
        size_t batch_size = num_elements / vocab_size;
        
        for (size_t batch = 0; batch < batch_size; batch++) {
            size_t offset = batch * vocab_size + idx;
            output[offset] -= value_h;
        }
    }
}

#if defined(__HAVE_BFLOAT__)
kernel void sub_at_indices_bf16(
    constant size_t &num_elements,
    constant size_t &vocab_size,
    constant uint32_t *indices,
    constant size_t &num_indices,
    constant float &value,
    device const bfloat *input,
    device bfloat *output,
    uint tid [[ thread_position_in_grid ]],
    uint tsize [[ threads_per_grid ]]
) {
    for (uint id = tid; id < num_elements; id += tsize) {
        output[id] = input[id];
    }
    
    threadgroup_barrier(mem_flags::mem_device);
    
    if (tid < num_indices) {
        uint32_t idx = indices[tid];
        bfloat value_bf = bfloat(value);
        
        size_t batch_size = num_elements / vocab_size;
        
        for (size_t batch = 0; batch < batch_size; batch++) {
            size_t offset = batch * vocab_size + idx;
            output[offset] -= value_bf;
        }
    }
}
#endif
