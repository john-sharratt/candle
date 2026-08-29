//! AOT-compiled CUDA kernels for the fork's inference engine, plus their FFI
//! bindings.
//!
//! `build.rs` invokes NVCC to compile each `.cu` source into PTX, keyed by a
//! SHA256 of the source tree so unchanged kernels are not recompiled; the
//! resulting PTX is embedded into the binary at compile time (no NVCC or
//! `.ptx` file is needed at runtime — `cargo build --features cuda` is
//! sufficient, see `make clean-ptx` to force a full rebuild). Each `pub mod`
//! below corresponds to a `src/<subdir>/` of `.cu` kernels plus an `api.rs`
//! Rust wrapper: `simple` (generic elementwise/reduce/indexing/conv ops used
//! by `candle-core::cuda_backend`), `quantized` (GGML-format quantized
//! matmul), `sampling` (batched logit-processing/sampling), `provenance`
//! (Binary Directional Provenance scan kernels for KV-chunk retrieval), and
//! `paged_decode`/`paged_prefill`/`paged_glue` (the paged, per-block-quantized
//! attention kernels backing the three-tier KV cache). `CHUNK_SIZE = 32` is
//! the shared Rust/CUDA block-size constant used throughout the paged and
//! quantized kernels.

/// Chunk size for paged attention kernels.
/// Must match CHUNK_SIZE in arena_table.cuh (compile-time constant = 32).
/// This value is used for GGML quantization alignment and fast bit-shift division.
pub const CHUNK_SIZE: i32 = 32;

// `BUILT_ARCHES` — the `sm_NN` list these archives carry SASS for, written by
// `build.rs` from the same constant the `-gencode` flags come from.
include!(concat!(env!("OUT_DIR"), "/built_arches.rs"));

/// Whether a device of compute capability `major.minor` has native SASS here.
///
/// A cubin built for `X.y` runs on `X.z` when `z >= y` — CUDA guarantees binary
/// compatibility forward across minor revisions, never backward and never
/// across a major revision. So sm_86 code covers 8.6, 8.7 and 8.9, while an
/// 8.0 device (A100) is *not* covered by it despite being the lower number.
///
/// There is no PTX in these archives, so this is the whole answer: false means
/// the device has nothing to run and nothing to JIT from.
pub fn has_kernel_image(major: u32, minor: u32) -> bool {
    BUILT_ARCHES
        .iter()
        .any(|&built| built / 10 == major && built % 10 <= minor)
}

pub mod simple;

#[path = "quantized/api.rs"]
pub mod quantized;

#[path = "sampling/api.rs"]
pub mod sampling;

#[path = "provenance/api.rs"]
pub mod provenance;

#[path = "paged-decode/api.rs"]
pub mod paged_decode;

#[path = "paged-prefill/api.rs"]
pub mod paged_prefill;

#[path = "paged-glue/api.rs"]
pub mod paged_glue;

#[path = "paged-latent/api.rs"]
pub mod paged_latent;

#[path = "delta-net/api.rs"]
pub mod delta_net;

#[cfg(test)]
mod built_arch_tests {
    use super::*;

    /// The fleet, and the cards that inherit its images by minor-forward
    /// compatibility. These must hold for whatever `KERNEL_ARCHES` contains, so
    /// they are asserted against `has_kernel_image` rather than a literal list.
    #[test]
    fn every_built_arch_covers_itself() {
        for &built in BUILT_ARCHES {
            assert!(
                has_kernel_image(built / 10, built % 10),
                "sm_{built} is built but reports no image"
            );
        }
    }

    /// `X.y` code runs on `X.z` for `z >= y`, never the reverse. The A100 is
    /// the case worth pinning: 8.0 is a *lower* minor than the 8.6 image, so it
    /// is not covered despite Ampere being "the same generation".
    #[test]
    fn coverage_runs_forward_across_minors_and_never_backward() {
        assert!(BUILT_ARCHES.contains(&86), "this test assumes sm_86 is built");
        assert!(has_kernel_image(8, 6));
        assert!(has_kernel_image(8, 7), "Jetson Orin inherits the 8.6 image");
        assert!(has_kernel_image(8, 9), "Ada is covered natively and by 8.6");
        assert!(!has_kernel_image(8, 0), "A100 is 8.0 — below the 8.6 image");
        assert!(!has_kernel_image(7, 5), "Turing is a different major");
    }

    /// A major version nobody built for has no image, whatever its minor.
    #[test]
    fn an_unbuilt_major_is_never_covered() {
        let majors: Vec<u32> = BUILT_ARCHES.iter().map(|s| s / 10).collect();
        for major in [7u32, 9, 10, 11] {
            if majors.contains(&major) {
                continue;
            }
            for minor in 0..10 {
                assert!(
                    !has_kernel_image(major, minor),
                    "SM {major}.{minor} reported an image from {BUILT_ARCHES:?}"
                );
            }
        }
    }
}
