//! Pointer-table layouts and address checks shared by the DeltaNet launchers.
//!
//! Short and decode tables carry a conv tail, a state and a timeline per span.
//! Long tables carry the entering and advanced buffers for each span, so the
//! prefill conv can read the entering tail while another block writes the
//! advance. The helpers check dtype and contiguity, and they reject a misaligned
//! base, since a misaligned state faults inside a kernel that cannot say which
//! operand it was.

use super::*;

pub(super) const MAX_HEAD_DIM: usize = 256;

/// Rows of a decode or short-span pointer table: conv tail, state, timeline.
/// The kernels read the first two and write only the third, so a step that
/// fails leaves every state as it entered.
pub(super) const TABLE_ROWS: usize = 3;

/// Rows of a long-span pointer table: conv tail in, conv tail out, state in,
/// state out — the entering buffers and the prefill scratch the advance is
/// written to, which `commit_wave` exchanges in.
pub(super) const LONG_TABLE_ROWS: usize = 4;

/// Rows of a fold table: conv tail, state, timeline, and the rows to fold.
pub(super) const FOLD_TABLE_ROWS: usize = 4;

/// Device address of `t`'s first element, checking dtype and contiguity.
///
/// Public because it is generic tensor plumbing rather than anything about the
/// mixer: the MTP head's kernel wrapper builds the same pointer tables from the
/// same checks, and a second transcription of "resolve, verify, hand the
/// address to a kernel" is a second place for the verification to be forgotten.
pub fn typed_ptr(t: &LiveTensor<'_>, dtype: DType, what: &str) -> Result<u64> {
    if t.dtype() != dtype {
        candle::bail!(
            "delta_net cuda: {what} must be {dtype:?}, got {:?}",
            t.dtype()
        );
    }
    if !t.is_contiguous() {
        candle::bail!("delta_net cuda: {what} must be contiguous");
    }
    let (storage, layout) = t.storage_and_layout();
    let cuda = match &*storage {
        Storage::Cuda(c) => c,
        _ => candle::bail!("delta_net cuda: {what} must be a CUDA tensor"),
    };
    let dev = match t.device() {
        Device::Cuda(d) => d.clone(),
        _ => candle::bail!("delta_net cuda: {what} must live on a CUDA device"),
    };
    let stream = dev.cuda_stream();
    let offset = layout.start_offset();
    // The guard is dropped before returning, as in every other launcher here:
    // the address stays valid because the storage outlives the launch, and the
    // stream ordering the guard would record is already implied by every op on
    // this device sharing one stream.
    let ptr = match dtype {
        DType::F32 => {
            let view = cuda.as_cuda_slice::<f32>()?.slice(offset..);
            let (ptr, _guard) = view.device_ptr(&stream);
            ptr
        }
        DType::I64 => {
            let view = cuda.as_cuda_slice::<i64>()?.slice(offset..);
            let (ptr, _guard) = view.device_ptr(&stream);
            ptr
        }
        DType::U32 => {
            let view = cuda.as_cuda_slice::<u32>()?.slice(offset..);
            let (ptr, _guard) = view.device_ptr(&stream);
            ptr
        }
        _ => candle::bail!("delta_net cuda: no pointer accessor for {dtype:?}"),
    };
    Ok(ptr)
}

pub fn f32_ptr(t: &LiveTensor<'_>, what: &str) -> Result<u64> {
    typed_ptr(t, DType::F32, what)
}

/// `ptr`, checked 16-byte aligned: the prefill passes read the state, the
/// Q|K rows and the scan transients four floats at a time, and a misaligned
/// base would fault inside a kernel that cannot say which operand it was.
pub fn aligned16(ptr: u64, what: &str) -> Result<u64> {
    if !ptr.is_multiple_of(16) {
        candle::bail!("delta_net cuda: {what} at {ptr:#x} is not 16-byte aligned");
    }
    Ok(ptr)
}

/// The address of a short span's timeline, 16-byte aligned (the kernels write
/// its rows four floats at a time).
pub(super) fn timeline_ptr(s: &DeltaNetSeq<'_>) -> Result<u64> {
    let Some(tl) = s.write.timeline() else {
        candle::bail!(
            "delta_net cuda: a {}-row span was handed a state to write, where every span \
             of at most {TIMELINE_ROWS} rows writes a timeline",
            s.len
        );
    };
    aligned16(f32_ptr(tl, "timeline")?, "timeline")
}
