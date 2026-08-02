//! One submodule per Metal op family, each exposing the `call_*` functions
//! re-exported here that `candle-core`'s `metal_backend` calls to encode and
//! dispatch a compute pass: `affine`/`binary`/`unary`/`ternary`/`cast`
//! (elementwise), `reduce`/`sort`/`multinomial`, `indexing`/`div_at_indices`/
//! `sub_at_indices`/`fill`, `convolution`, `mlx_gemm`/`quantized` (dense and
//! GGML-quantized matmul), and `sdpa` (scaled dot-product attention).
//! `macros` holds the shared `EncoderParam`/argument-binding boilerplate used
//! across all of them.

pub mod affine;
pub mod binary;
pub mod cast;
pub mod convolution;
pub mod div_at_indices;
pub mod fill;
pub mod indexing;
mod macros;
pub mod mlx_gemm;
pub mod multinomial;
pub mod quantized;
pub mod random;
pub mod reduce;
pub mod sdpa;
pub mod sort;
pub mod sub_at_indices;
pub mod ternary;
pub mod unary;

pub use affine::*;
pub use binary::{call_binary_contiguous, call_binary_strided};
pub use cast::{call_cast_contiguous, call_cast_strided};
pub use convolution::*;
pub use div_at_indices::*;
pub use fill::*;
pub use indexing::*;
pub use mlx_gemm::{call_mlx_gemm, GemmDType};
pub use multinomial::MULTINOMIAL;
pub use quantized::{call_quantized_matmul_mm_t, call_quantized_matmul_mv_t, GgmlDType};
pub use random::*;
pub use reduce::*;
pub use sdpa::{call_sdpa_full, call_sdpa_vector, call_sdpa_vector_2pass, SdpaDType};
pub use sort::{call_arg_sort, call_mlx_arg_sort};
pub use sub_at_indices::*;
pub use ternary::call_where_cond;
pub use unary::*;
