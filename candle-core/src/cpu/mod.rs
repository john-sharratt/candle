//! SIMD dot-product/reduction primitives shared by the CPU backend.
//!
//! Defines the `Cpu`/`CpuF16`/`CpuBF16` traits (load/fma/reduce over a fixed
//! `ARR`-wide register array) and re-exports whichever architecture module
//! (`avx`, `neon`, `simd128`) implements `CurrentCpu`/`CurrentCpuF16`/
//! `CurrentCpuBF16` for the target, selected purely by `cfg(target_feature)`.
//! `vec_dot_f32`/`vec_dot_f16`/`vec_dot_bf16`/`vec_sum` are the row-wise
//! kernels `cpu_backend`'s `MatMul`/`ReduceSum` call into; each has a scalar
//! fallback compiled in when no SIMD feature is enabled.

pub mod erf;
pub mod kernels;

#[allow(unused)]
trait Cpu<const ARR: usize> {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;

    fn n() -> usize;
    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f32) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f32, a: Self::Unit);
}

#[allow(unused)]
trait CpuF16<const ARR: usize> {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;

    fn n() -> usize;
    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const f16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut f16, a: Self::Unit);
}

#[allow(unused)]
trait CpuBF16<const ARR: usize> {
    type Unit;
    type Array;
    const STEP: usize;
    const EPR: usize;

    fn n() -> usize;
    unsafe fn zero() -> Self::Unit;
    unsafe fn zero_array() -> Self::Array;
    unsafe fn load(mem_addr: *const bf16) -> Self::Unit;
    unsafe fn vec_add(a: Self::Unit, b: Self::Unit) -> Self::Unit;
    unsafe fn vec_fma(a: Self::Unit, b: Self::Unit, c: Self::Unit) -> Self::Unit;
    unsafe fn vec_reduce(x: Self::Array, y: *mut f32);
    unsafe fn from_f32(v: f32) -> Self::Unit;
    unsafe fn vec_store(mem_addr: *mut bf16, a: Self::Unit);
}

use half::{bf16, f16};

// x86: the AVX kernels are always compiled, each with its own
// `target_feature`, and run where the CPU reports AVX and F16C; the binary's
// own baseline is below AVX (`.cargo/config.toml`).
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod avx;

#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub mod simd128;
#[cfg(target_arch = "wasm32")]
#[cfg(target_feature = "simd128")]
pub use simd128::CurrentCpu;

#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(any(target_arch = "arm", target_arch = "aarch64"))]
#[cfg(target_feature = "neon")]
pub use neon::CurrentCpu;

/// `*c = Σ a[i]·b[i]` over `k` f32s.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    if avx::available() {
        return avx::vec_dot_f32(a_row, b_row, c, k);
    }
    scalar::vec_dot_f32(a_row, b_row, c, k)
}

/// `*b = Σ row[i]` over `k` f32s.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    if avx::available() {
        return avx::vec_sum(row, b, k);
    }
    scalar::vec_sum(row, b, k)
}

/// `*c = Σ a[i]·b[i]` over `k` f16s, accumulated in f32.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
    if avx::available() {
        return avx::vec_dot_f16(a_row, b_row, c, k);
    }
    scalar::vec_dot_f16(a_row, b_row, c, k)
}

/// `*c = Σ a[i]·b[i]` over `k` bf16s, accumulated in f32.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[inline]
pub(crate) unsafe fn vec_dot_bf16(a_row: *const bf16, b_row: *const bf16, c: *mut f32, k: usize) {
    if avx::available() {
        return avx::vec_dot_bf16(a_row, b_row, c, k);
    }
    scalar::vec_dot_bf16(a_row, b_row, c, k)
}

#[cfg(all(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    any(target_feature = "neon", target_feature = "simd128")
))]
#[inline(always)]
pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
    let np = k & !(CurrentCpu::STEP - 1);

    let mut sum = CurrentCpu::zero_array();
    let mut ax = CurrentCpu::zero_array();
    let mut ay = CurrentCpu::zero_array();

    for i in (0..np).step_by(CurrentCpu::STEP) {
        for j in 0..CurrentCpu::n() {
            ax[j] = CurrentCpu::load(a_row.add(i + j * CurrentCpu::EPR));
            ay[j] = CurrentCpu::load(b_row.add(i + j * CurrentCpu::EPR));

            sum[j] = CurrentCpu::vec_fma(sum[j], ax[j], ay[j]);
        }
    }

    CurrentCpu::vec_reduce(sum, c);

    // leftovers
    for i in np..k {
        *c += *a_row.add(i) * (*b_row.add(i));
    }
}

#[cfg(all(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    any(target_feature = "neon", target_feature = "simd128")
))]
#[inline(always)]
pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    let np = k & !(CurrentCpu::STEP - 1);

    let mut sum = CurrentCpu::zero_array();
    let mut x = CurrentCpu::zero_array();

    for i in (0..np).step_by(CurrentCpu::STEP) {
        for j in 0..CurrentCpu::n() {
            x[j] = CurrentCpu::load(row.add(i + j * CurrentCpu::EPR));
            sum[j] = CurrentCpu::vec_add(sum[j], x[j]);
        }
    }

    CurrentCpu::vec_reduce(sum, b);

    // leftovers
    for i in np..k {
        *b += *row.add(i)
    }
}

#[cfg(all(
    not(any(target_arch = "x86", target_arch = "x86_64")),
    not(any(target_feature = "neon", target_feature = "simd128"))
))]
pub(crate) use scalar::{vec_dot_f32, vec_sum};

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
pub(crate) use scalar::{vec_dot_bf16, vec_dot_f16};

/// The portable loops: what a CPU without a vector path runs.
#[allow(dead_code)]
mod scalar {
    use half::{bf16, f16};

    #[inline(always)]
    pub(crate) unsafe fn vec_dot_f32(a_row: *const f32, b_row: *const f32, c: *mut f32, k: usize) {
        for i in 0..k {
            *c += *a_row.add(i) * (*b_row.add(i));
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
        *b = 0f32;
        for i in 0..k {
            *b += *row.add(i)
        }
    }

    #[inline(always)]
    pub(crate) unsafe fn vec_dot_f16(a_row: *const f16, b_row: *const f16, c: *mut f32, k: usize) {
        let mut sum = 0.0;
        for i in 0..k {
            sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        }
        *c = sum;
    }

    #[inline(always)]
    pub(crate) unsafe fn vec_dot_bf16(
        a_row: *const bf16,
        b_row: *const bf16,
        c: *mut f32,
        k: usize,
    ) {
        let mut sum = 0.0;
        for i in 0..k {
            sum += (*a_row.add(i)).to_f32() * (*b_row.add(i)).to_f32();
        }
        *c = sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32s(n: usize, seed: f32) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32 * 0.37 + seed).sin() * 3.0).round() / 4.0)
            .collect()
    }

    /// Each kernel against the plain sum of products, at widths that cover a
    /// whole step, leftovers only, and both. The values are multiples of 1/4
    /// with small sums, so every order of addition is exact and the vector and
    /// scalar paths must agree to the bit.
    #[test]
    fn every_kernel_equals_the_plain_sum() {
        for k in [0, 5, 32, 33, 100, 257] {
            let (a, b) = (f32s(k, 0.1), f32s(k, 1.3));
            let want: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();

            let mut got = 0.0;
            unsafe { vec_dot_f32(a.as_ptr(), b.as_ptr(), &mut got, k) };
            assert_eq!(got, want, "f32 dot, k={k}");

            let mut s = 0.0;
            unsafe { vec_sum(a.as_ptr(), &mut s, k) };
            assert_eq!(s, a.iter().sum::<f32>(), "sum, k={k}");

            let (ah, bh): (Vec<f16>, Vec<f16>) = (
                a.iter().map(|&v| f16::from_f32(v)).collect(),
                b.iter().map(|&v| f16::from_f32(v)).collect(),
            );
            let mut got = 0.0;
            unsafe { vec_dot_f16(ah.as_ptr(), bh.as_ptr(), &mut got, k) };
            assert_eq!(got, want, "f16 dot, k={k}");

            let (ab, bb): (Vec<bf16>, Vec<bf16>) = (
                a.iter().map(|&v| bf16::from_f32(v)).collect(),
                b.iter().map(|&v| bf16::from_f32(v)).collect(),
            );
            let mut got = 0.0;
            unsafe { vec_dot_bf16(ab.as_ptr(), bb.as_ptr(), &mut got, k) };
            assert_eq!(got, want, "bf16 dot, k={k}");
        }
    }
}
