//! Expert weight handles.
//!
//! The expert forward itself runs on the device (`dispatch`): bucketize,
//! gather, grouped GEMMs over the live pointer table, SwiGLU and the
//! deterministic scatter. What lives here is the small piece of weight plumbing
//! other modules share.

// Re-export QMatMul so other submodules can reference it through `compute::QMatMul`.
pub(crate) use crate::models::quantized_matmul::QMatMul;

/// Extract the CUDA data pointer from a QMatMul wrapper.
///
/// Returns `(device_ptr, shape, ggml_dtype)` for the underlying quantized tensor,
/// or an error if the tensor is not a CUDA QTensor.
#[cfg(feature = "cuda")]
pub(crate) fn extract_weight_info(
    qmm: &QMatMul,
) -> candle::Result<(u64, candle::Shape, candle::quantized::GgmlDType)> {
    match qmm.inner() {
        candle::quantized::QMatMul::QTensor(qt) => {
            let ptr = qt
                .cuda_data_ptr()
                .ok_or_else(|| candle::Error::Msg("expected CUDA QTensor".into()))?;
            Ok((ptr, qt.shape().clone(), qt.dtype()))
        }
        _ => Err(candle::Error::Msg(
            "expert weight handle requires the QTensor variant".into(),
        )),
    }
}
