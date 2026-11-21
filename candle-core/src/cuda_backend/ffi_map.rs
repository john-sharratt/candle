//! Maps candle's dtypes and operations to the codes the CUDA FFI dispatchers
//! take.
//!
//! Each function converts one tensor-level identity, such as a dtype, a unary,
//! binary or comparison op, or an index or scatter type, into the enum value the
//! kernel dispatcher expects. Where a dtype has no kernel the function returns
//! `None` rather than choosing a neighbour: scatter and repeat-penalty support
//! only f32, f64, f16 and bf16, and convolution refuses I64 and F8E4M3. The
//! caller then sees the unsupported pair and reports it, instead of launching
//! the wrong kernel.

use super::*;

pub fn kernel_name<T: WithDType>(root: &str) -> String {
    let dtype = T::DTYPE.as_str();
    format!("{root}_{dtype}")
}

/// Convert candle DType to AffineDType for FFI dispatcher
pub(super) fn dtype_to_affine_dtype(dtype: DType) -> i32 {
    use kernels::simple::affine::AffineDType;
    match dtype {
        DType::F32 => AffineDType::F32 as i32,
        DType::F64 => AffineDType::F64 as i32,
        DType::F16 => AffineDType::F16 as i32,
        DType::BF16 => AffineDType::BF16 as i32,
        DType::F8E4M3 => AffineDType::F8E4M3 as i32,
        DType::U8 => AffineDType::U8 as i32,
        DType::U32 => AffineDType::U32 as i32,
        DType::I64 => AffineDType::I64 as i32,
    }
}

/// Convert candle DType to UnaryDType for FFI dispatcher
pub(super) fn dtype_to_unary_dtype(dtype: DType) -> i32 {
    use kernels::simple::unary::UnaryDType;
    match dtype {
        DType::F32 => UnaryDType::F32 as i32,
        DType::F64 => UnaryDType::F64 as i32,
        DType::F16 => UnaryDType::F16 as i32,
        DType::BF16 => UnaryDType::BF16 as i32,
        DType::F8E4M3 => UnaryDType::F8E4M3 as i32,
        DType::U8 => UnaryDType::U8 as i32,
        DType::U32 => UnaryDType::U32 as i32,
        DType::I64 => UnaryDType::I64 as i32,
    }
}

/// Convert candle DType to FastReduceDType for FFI dispatcher
pub(super) fn dtype_to_fast_reduce_dtype(dtype: DType) -> i32 {
    use kernels::simple::reduce::FastReduceDType;
    match dtype {
        DType::F32 => FastReduceDType::F32 as i32,
        DType::F64 => FastReduceDType::F64 as i32,
        DType::F16 => FastReduceDType::F16 as i32,
        DType::BF16 => FastReduceDType::BF16 as i32,
        DType::U32 => FastReduceDType::U32 as i32,
        DType::I64 => FastReduceDType::I64 as i32,
        DType::U8 => FastReduceDType::U8 as i32,
        DType::F8E4M3 => FastReduceDType::F8E4M3 as i32,
    }
}

/// Convert candle DType to IndexingDataDType for FFI dispatcher
pub(super) fn dtype_to_indexing_data_dtype(dtype: DType) -> i32 {
    use kernels::simple::indexing::IndexingDataDType;
    match dtype {
        DType::F32 => IndexingDataDType::F32 as i32,
        DType::F64 => IndexingDataDType::F64 as i32,
        DType::U8 => IndexingDataDType::U8 as i32,
        DType::U32 => IndexingDataDType::U32 as i32,
        DType::I64 => IndexingDataDType::I64 as i32,
        DType::F16 => IndexingDataDType::F16 as i32,
        DType::BF16 => IndexingDataDType::BF16 as i32,
        DType::F8E4M3 => IndexingDataDType::F8E4M3 as i32,
    }
}

/// Convert index storage dtype to IndexDType for FFI dispatcher
pub(super) fn storage_to_index_dtype(slice: &CudaStorageSlice) -> Option<i32> {
    use kernels::simple::indexing::IndexDType;
    match slice {
        CudaStorageSlice::I64(_) => Some(IndexDType::I64 as i32),
        CudaStorageSlice::U32(_) => Some(IndexDType::U32 as i32),
        CudaStorageSlice::U8(_) => Some(IndexDType::U8 as i32),
        _ => None,
    }
}

/// Convert candle DType to ScatterDType for FFI dispatcher
/// Returns None for unsupported dtypes (only f32, f64, f16, bf16 supported)
pub(super) fn dtype_to_scatter_dtype(dtype: DType) -> Option<i32> {
    use kernels::simple::scatter_op::ScatterDType;
    match dtype {
        DType::F32 => Some(ScatterDType::F32 as i32),
        DType::F64 => Some(ScatterDType::F64 as i32),
        DType::F16 => Some(ScatterDType::F16 as i32),
        DType::BF16 => Some(ScatterDType::BF16 as i32),
        _ => None,
    }
}

/// Convert candle DType to RepeatPenaltyDType for FFI dispatcher
/// Returns None for unsupported dtypes (only f32, f64, f16, bf16 supported)
pub(super) fn dtype_to_repeat_penalty_dtype(dtype: DType) -> Option<i32> {
    use kernels::simple::repeat_penalty::RepeatPenaltyDType;
    match dtype {
        DType::F32 => Some(RepeatPenaltyDType::F32 as i32),
        DType::F64 => Some(RepeatPenaltyDType::F64 as i32),
        DType::F16 => Some(RepeatPenaltyDType::F16 as i32),
        DType::BF16 => Some(RepeatPenaltyDType::BF16 as i32),
        _ => None,
    }
}

/// Map kernel name to UnaryOp enum value for FFI dispatcher
pub(super) fn kernel_name_to_unary_op(kernel: &str) -> Option<i32> {
    use kernels::simple::unary::UnaryOp;
    match kernel {
        "ucopy" => Some(UnaryOp::Copy as i32),
        "uneg" => Some(UnaryOp::Neg as i32),
        "urecip" => Some(UnaryOp::Recip as i32),
        "uexp" => Some(UnaryOp::Exp as i32),
        "ulog" => Some(UnaryOp::Log as i32),
        "usin" => Some(UnaryOp::Sin as i32),
        "ucos" => Some(UnaryOp::Cos as i32),
        "utanh" => Some(UnaryOp::Tanh as i32),
        "uerf" => Some(UnaryOp::Erf as i32),
        "uceil" => Some(UnaryOp::Ceil as i32),
        "ufloor" => Some(UnaryOp::Floor as i32),
        "uround" => Some(UnaryOp::Round as i32),
        "unormcdf" => Some(UnaryOp::Normcdf as i32),
        "uabs" => Some(UnaryOp::Abs as i32),
        "usqr" => Some(UnaryOp::Sqr as i32),
        "usqrt" => Some(UnaryOp::Sqrt as i32),
        "ugelu" => Some(UnaryOp::Gelu as i32),
        "ugelu_erf" => Some(UnaryOp::GeluErf as i32),
        "urelu" => Some(UnaryOp::Relu as i32),
        "usilu" => Some(UnaryOp::Silu as i32),
        "usign" => Some(UnaryOp::Sign as i32),
        "usigmoid" => Some(UnaryOp::Sigmoid as i32),
        _ => None,
    }
}

/// Convert condition storage dtype to WhereCondDType for FFI dispatcher
pub(super) fn storage_to_where_cond_dtype(slice: &CudaStorageSlice) -> Option<i32> {
    use kernels::simple::ternary::WhereCondDType;
    match slice {
        CudaStorageSlice::I64(_) => Some(WhereCondDType::I64 as i32),
        CudaStorageSlice::U32(_) => Some(WhereCondDType::U32 as i32),
        CudaStorageSlice::U8(_) => Some(WhereCondDType::U8 as i32),
        _ => None,
    }
}

/// Convert candle DType to WhereDataDType for FFI dispatcher
pub(super) fn dtype_to_where_data_dtype(dtype: DType) -> i32 {
    use kernels::simple::ternary::WhereDataDType;
    match dtype {
        DType::F32 => WhereDataDType::F32 as i32,
        DType::F64 => WhereDataDType::F64 as i32,
        DType::U8 => WhereDataDType::U8 as i32,
        DType::U32 => WhereDataDType::U32 as i32,
        DType::I64 => WhereDataDType::I64 as i32,
        DType::F16 => WhereDataDType::F16 as i32,
        DType::BF16 => WhereDataDType::BF16 as i32,
        DType::F8E4M3 => WhereDataDType::F8E4M3 as i32,
    }
}

/// Convert candle DType to ConvDType for FFI dispatcher
/// Returns None for I64 and F8E4M3 which are not supported by conv operations
pub(super) fn dtype_to_conv_dtype(dtype: DType) -> Option<i32> {
    use kernels::simple::conv::ConvDType;
    match dtype {
        DType::F32 => Some(ConvDType::F32 as i32),
        DType::F64 => Some(ConvDType::F64 as i32),
        DType::F16 => Some(ConvDType::F16 as i32),
        DType::BF16 => Some(ConvDType::BF16 as i32),
        DType::U8 => Some(ConvDType::U8 as i32),
        DType::U32 => Some(ConvDType::U32 as i32),
        DType::I64 => None,    // Not supported by conv operations
        DType::F8E4M3 => None, // Not supported by conv operations
    }
}

/// Convert candle DType to BinaryDType for FFI dispatcher
pub(super) fn dtype_to_binary_dtype(dtype: DType) -> i32 {
    use kernels::simple::binary::BinaryDType;
    match dtype {
        DType::F32 => BinaryDType::F32 as i32,
        DType::F64 => BinaryDType::F64 as i32,
        DType::U8 => BinaryDType::U8 as i32,
        DType::U32 => BinaryDType::U32 as i32,
        DType::I64 => BinaryDType::I64 as i32,
        DType::F16 => BinaryDType::F16 as i32,
        DType::BF16 => BinaryDType::BF16 as i32,
        DType::F8E4M3 => BinaryDType::F8E4M3 as i32,
    }
}

/// Convert BinaryInplaceOp to FFI enum value for in-place binary dispatcher
pub(super) fn binary_inplace_op_to_ffi(op: crate::op::BinaryInplaceOp) -> i32 {
    use kernels::simple::binary::BinaryInplaceOp as FFIOp;
    match op {
        crate::op::BinaryInplaceOp::Add => FFIOp::Add as i32,
        crate::op::BinaryInplaceOp::Sub => FFIOp::Sub as i32,
        crate::op::BinaryInplaceOp::Mul => FFIOp::Mul as i32,
        crate::op::BinaryInplaceOp::Div => FFIOp::Div as i32,
        crate::op::BinaryInplaceOp::Min => FFIOp::Min as i32,
        crate::op::BinaryInplaceOp::Max => FFIOp::Max as i32,
    }
}

/// Map kernel name to BinaryArithOp enum value for FFI dispatcher
pub(super) fn kernel_name_to_binary_arith_op(kernel: &str) -> Option<i32> {
    use kernels::simple::binary::BinaryArithOp;
    match kernel {
        "badd" => Some(BinaryArithOp::Add as i32),
        "bdiv" => Some(BinaryArithOp::Div as i32),
        "bmul" => Some(BinaryArithOp::Mul as i32),
        "bsub" => Some(BinaryArithOp::Sub as i32),
        "bminimum" => Some(BinaryArithOp::Minimum as i32),
        "bmaximum" => Some(BinaryArithOp::Maximum as i32),
        _ => None,
    }
}

/// Map CmpOp to binary comparison op enum value for FFI dispatcher
pub(super) fn cmp_op_to_binary_cmp(op: CmpOp) -> i32 {
    use kernels::simple::binary::BinaryCmpOp;
    match op {
        CmpOp::Eq => BinaryCmpOp::Eq as i32,
        CmpOp::Ne => BinaryCmpOp::Ne as i32,
        CmpOp::Lt => BinaryCmpOp::Lt as i32,
        CmpOp::Le => BinaryCmpOp::Le as i32,
        CmpOp::Gt => BinaryCmpOp::Gt as i32,
        CmpOp::Ge => BinaryCmpOp::Ge as i32,
    }
}
