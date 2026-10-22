//! Useful functions for checking features.
use std::str::FromStr;

pub fn get_num_threads() -> usize {
    // Respond to the same environment variable as rayon.
    match std::env::var("RAYON_NUM_THREADS")
        .ok()
        .and_then(|s| usize::from_str(&s).ok())
    {
        Some(x) if x > 0 => x,
        Some(_) | None => num_cpus::get(),
    }
}

pub fn has_accelerate() -> bool {
    cfg!(feature = "accelerate")
}

pub fn has_mkl() -> bool {
    cfg!(feature = "mkl")
}

pub fn cuda_is_available() -> bool {
    cfg!(feature = "cuda")
}

pub fn metal_is_available() -> bool {
    cfg!(feature = "metal")
}

/// Whether the quantized kernels run their AVX2 path on this CPU — a runtime
/// answer, since the binary's own baseline is below AVX2.
pub fn with_avx() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return crate::quantized::avx::available();
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    false
}

pub fn with_neon() -> bool {
    cfg!(target_feature = "neon")
}

pub fn with_simd128() -> bool {
    cfg!(target_feature = "simd128")
}

/// Whether this CPU has F16C — asked at runtime, for the same reason.
pub fn with_f16c() -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    return is_x86_feature_detected!("f16c");
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    false
}
