use crate::linear_split;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{set_params, Buffer, Device, Kernels, MetalKernelError, Source};
use objc2_metal::MTLResourceUsage;

pub fn call_sub_at_indices(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: &'static str,
    num_elements: usize,
    vocab_size: usize,
    indices: &Buffer,
    num_indices: usize,
    value: f32,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::SubAtIndices, name)?;

    let encoder = ep.encoder();
    let encoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            num_elements,
            vocab_size,
            indices,
            num_indices,
            value,
            &input,
            output
        )
    );

    let thread_count = num_elements.max(num_indices);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, thread_count);
    encoder.use_resource(input.buffer, MTLResourceUsage::Read);
    encoder.use_resource(indices, MTLResourceUsage::Read);
    encoder.use_resource(output, MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}
