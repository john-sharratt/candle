//! The CPU expert kernel (`docs/cpu_expert_compute.md` §5.10): int8
//! activations against KO-quantized expert weights read in place.
//!
//! | Module | Concern |
//! |---|---|
//! | [`act`] | q8 activation tiles: quantizing, and reading the device's q8a1024 operand |
//! | [`ko`] | the KO weight layout, its unpack order, and the scalar reference matvec |
//! | [`simd`] | the matvec by instruction-set tier, byte-equal to the reference |
//! | `simd512` | the AVX-512 VNNI tier: 64-byte unpack, `zmm` dots |
//! | [`expert`] | one expert: gate, up, SwiGLU, down |

pub mod act;
pub mod expert;
pub mod ko;
pub mod simd;
#[cfg(target_arch = "x86_64")]
mod simd512;
