//! The quantized-tensor layer: the GGML and GGUF formats, the matmul operator
//! over them, and the storage that each one lives in.
//!
//! A quantized weight is a tensor of blocks in one dtype, either owned or leased.
//! It is repacked once at load into the layout that the int8 tensor-core path or
//! the float path reads. The GGUF file codes and the in-workspace dtype numbers
//! are translated at a single boundary. Devices differ in what they support, so
//! each has its own storage, and non-CUDA builds refuse the CUDA-only paths by
//! name. A wrong dtype or layout here produces wrong numbers rather than an
//! error, which is why the constructors and tests check them exactly.
//!
//! - `ggml_dtype`: the dtype enumeration, bits per weight, and GGUF file codes.
//! - `int8_mode`: the int8 numeric modes and the sizing rule that picks one.
//! - `live_tensor`: quantized tensors with a stated memory lifetime.
//! - `qmatmul`: the matmul operator and the load-time repacking of weights.
//! - `qmatmul_forward`: the forward passes, each stored at the width its consumer needs.
//! - `qstorage`: device-or-host storage with ranged byte reads and writes.
//! - `qtensor`: the constructors and the float-to-quantized path.
//!
//! - `tests_ggml_dtype`, `tests_int8_mode`: tests for the dtype tables and the
//!   int8 sizing rule.

use crate::{Context, CpuStorage, DType, Device, LiveTensor, Result, Shape, Storage, Tensor};
use k_quants::*;
use std::borrow::Cow;
use std::marker::PhantomData;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
pub mod avx;
pub mod decode_rows;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod gguf_writer;
#[cfg(test)]
mod int8_block_codec_tests;
pub mod int8_matmul_mode;
pub mod int8_split_k;
pub mod k_quants;
pub mod ko_quant;
#[cfg(feature = "metal")]
pub mod metal;
pub mod prepare;

#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}

#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(not(feature = "cuda"))]
mod dummy_pinned_staging;
#[cfg(feature = "cuda")]
pub mod pinned_staging;
#[cfg(feature = "cuda")]
mod resident_ring;
#[cfg(feature = "cuda")]
pub mod table_ring;

#[cfg(not(feature = "cuda"))]
pub mod pinned_staging {
    pub use super::dummy_pinned_staging::*;
}

#[cfg(not(feature = "cuda"))]
pub mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(feature = "cuda")]
pub use cuda::cuda_flush_l2;
pub use cuda::get_dispatch_info;
#[cfg(feature = "cuda")]
pub use cuda::grouped_matmul_gemx;
pub use cuda::set_force_dmmv;
#[cfg(feature = "cuda")]
pub use cuda::{
    alloc_host_mapped, get_total_vram_device0, get_vram_info, register_mmap_cuda, HostMappedAlloc,
    MmapRegistration,
};
#[cfg(feature = "cuda")]
pub use cuda::{
    load_repacked, load_repacked_into, load_repacked_on_stream, repack_gemx_to_host,
    repack_to_host, repacked_size_bytes, view_repacked,
};

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
mod warp_mirror;
use half::{bf16, f16};
#[cfg(feature = "cuda")]
use int8_split_k::DensePlan;

pub use k_quants::GgmlType;

mod ggml_dtype;
mod int8_mode;
mod live_tensor;
mod qmatmul;
mod qmatmul_forward;
mod qstorage;
mod qtensor;
#[cfg(test)]
mod tests_ggml_dtype;
#[cfg(test)]
mod tests_int8_mode;

pub use ggml_dtype::*;
pub use int8_mode::*;
pub use live_tensor::*;
pub use qmatmul::*;
pub use qstorage::*;
