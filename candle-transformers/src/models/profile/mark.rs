//! The opaque host timestamp the accumulator measures against.
//!
//! `Instant` under `profile`, `()` without it — which is what lets every caller
//! keep the same spelling in both builds and pay nothing in the second.

/// Opaque timestamp. `Instant` when `profile` is enabled, `()` otherwise.
#[cfg(feature = "profile")]
pub type ProfileMark = std::time::Instant;

/// Opaque timestamp. `Instant` when `profile` is enabled, `()` otherwise.
#[cfg(not(feature = "profile"))]
pub type ProfileMark = ();

/// Capture the current timestamp. Compiles to nothing when profiling is
/// disabled.
#[cfg(feature = "profile")]
#[inline(always)]
pub fn profile_now() -> ProfileMark {
    std::time::Instant::now()
}

/// No-op when profiling is disabled.
#[cfg(not(feature = "profile"))]
#[inline(always)]
pub fn profile_now() -> ProfileMark {}
