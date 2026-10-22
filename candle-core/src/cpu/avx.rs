//! The CPU backend's f32 / f16 / bf16 row kernels in 256-bit AVX, chosen at
//! runtime.
//!
//! The binary is built for a CPU level every cluster member has
//! (`.cargo/config.toml`), below AVX, so these are compiled with their own
//! `target_feature` and run only where the CPU reports AVX and F16C
//! ([`available`]); everywhere else `super` runs the scalar loops. Each kernel
//! keeps the accumulation order of the scalar-free path it replaced — four
//! 8-lane accumulators, reduced pairwise — so its results are unchanged.

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
use std::sync::LazyLock;

use half::{bf16, f16};

/// Elements one loop step covers: four accumulators of eight lanes.
const STEP: usize = 32;
/// Lanes per accumulator.
const EPR: usize = 8;
/// Accumulators.
const ARR: usize = STEP / EPR;

/// Whether this CPU runs these kernels — asked once.
static AVAILABLE: LazyLock<bool> =
    LazyLock::new(|| is_x86_feature_detected!("avx") && is_x86_feature_detected!("f16c"));

/// Whether this CPU runs the AVX kernels.
pub(super) fn available() -> bool {
    *AVAILABLE
}

/// The four accumulators summed as `(0+1) + (2+3)`, then across lanes — the
/// f32 kernels' reduction.
#[target_feature(enable = "avx")]
unsafe fn reduce_adjacent(mut x: [__m256; ARR]) -> f32 {
    x[0] = _mm256_add_ps(x[0], x[1]);
    x[2] = _mm256_add_ps(x[2], x[3]);
    x[0] = _mm256_add_ps(x[0], x[2]);
    lanes(x[0])
}

/// The four accumulators summed as `(0+2) + (1+3)`, then across lanes — the
/// half-precision kernels' reduction.
#[target_feature(enable = "avx")]
unsafe fn reduce_halves(mut x: [__m256; ARR]) -> f32 {
    x[0] = _mm256_add_ps(x[0], x[2]);
    x[1] = _mm256_add_ps(x[1], x[3]);
    x[0] = _mm256_add_ps(x[0], x[1]);
    lanes(x[0])
}

/// The eight lanes of `x` summed.
#[target_feature(enable = "avx")]
unsafe fn lanes(x: __m256) -> f32 {
    let t0 = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
    let t1 = _mm_hadd_ps(t0, t0);
    _mm_cvtss_f32(_mm_hadd_ps(t1, t1))
}

/// Eight f16 values widened to f32.
#[target_feature(enable = "avx,f16c")]
unsafe fn load_f16(p: *const f16) -> __m256 {
    _mm256_cvtph_ps(_mm_loadu_si128(p.cast()))
}

/// Eight bf16 values widened to f32: a bf16 is the high half of an f32, so
/// each one moves up 16 bits into a lane whose low half is zero.
#[target_feature(enable = "avx")]
unsafe fn load_bf16(p: *const bf16) -> __m256 {
    let raw = _mm_loadu_si128(p.cast());
    let zero = _mm_setzero_si128();
    let lo = _mm_castsi128_ps(_mm_unpacklo_epi16(zero, raw));
    let hi = _mm_castsi128_ps(_mm_unpackhi_epi16(zero, raw));
    _mm256_set_m128(hi, lo)
}

/// `*c = Σ a[i]·b[i]` over `k` f32s.
///
/// # Safety
///
/// [`available`] must hold, and `a` and `b` must be readable for `k` values.
#[target_feature(enable = "avx")]
pub(super) unsafe fn vec_dot_f32(a: *const f32, b: *const f32, c: *mut f32, k: usize) {
    let np = k & !(STEP - 1);
    let mut sum = [_mm256_setzero_ps(); ARR];
    for i in (0..np).step_by(STEP) {
        for (j, s) in sum.iter_mut().enumerate() {
            let ax = _mm256_loadu_ps(a.add(i + j * EPR));
            let ay = _mm256_loadu_ps(b.add(i + j * EPR));
            *s = _mm256_add_ps(_mm256_mul_ps(ax, ay), *s);
        }
    }
    *c = reduce_adjacent(sum);
    for i in np..k {
        *c += *a.add(i) * *b.add(i);
    }
}

/// `*b = Σ row[i]` over `k` f32s.
///
/// # Safety
///
/// [`available`] must hold, and `row` must be readable for `k` values.
#[target_feature(enable = "avx")]
pub(super) unsafe fn vec_sum(row: *const f32, b: *mut f32, k: usize) {
    let np = k & !(STEP - 1);
    let mut sum = [_mm256_setzero_ps(); ARR];
    for i in (0..np).step_by(STEP) {
        for (j, s) in sum.iter_mut().enumerate() {
            *s = _mm256_add_ps(*s, _mm256_loadu_ps(row.add(i + j * EPR)));
        }
    }
    *b = reduce_adjacent(sum);
    for i in np..k {
        *b += *row.add(i);
    }
}

/// `*c = Σ a[i]·b[i]` over `k` f16s, accumulated in f32.
///
/// # Safety
///
/// [`available`] must hold, and `a` and `b` must be readable for `k` values.
#[target_feature(enable = "avx,f16c")]
pub(super) unsafe fn vec_dot_f16(a: *const f16, b: *const f16, c: *mut f32, k: usize) {
    let np = k & !(STEP - 1);
    let mut sum = [_mm256_setzero_ps(); ARR];
    for i in (0..np).step_by(STEP) {
        for (j, s) in sum.iter_mut().enumerate() {
            let ax = load_f16(a.add(i + j * EPR));
            let ay = load_f16(b.add(i + j * EPR));
            *s = _mm256_add_ps(_mm256_mul_ps(ax, ay), *s);
        }
    }
    let mut sumf = reduce_halves(sum);
    for i in np..k {
        sumf += (*a.add(i)).to_f32() * (*b.add(i)).to_f32();
    }
    *c = sumf;
}

/// `*c = Σ a[i]·b[i]` over `k` bf16s, accumulated in f32.
///
/// # Safety
///
/// [`available`] must hold, and `a` and `b` must be readable for `k` values.
#[target_feature(enable = "avx")]
pub(super) unsafe fn vec_dot_bf16(a: *const bf16, b: *const bf16, c: *mut f32, k: usize) {
    let np = k & !(STEP - 1);
    let mut sum = [_mm256_setzero_ps(); ARR];
    for i in (0..np).step_by(STEP) {
        for (j, s) in sum.iter_mut().enumerate() {
            let ax = load_bf16(a.add(i + j * EPR));
            let ay = load_bf16(b.add(i + j * EPR));
            *s = _mm256_add_ps(_mm256_mul_ps(ax, ay), *s);
        }
    }
    let mut sumf = reduce_halves(sum);
    for i in np..k {
        sumf += (*a.add(i)).to_f32() * (*b.add(i)).to_f32();
    }
    *c = sumf;
}
