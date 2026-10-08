//! A target's fingerprint: a number that changes when what the target holds
//! changes, and never otherwise.
//!
//! Fingerprints are saved with the world, so they must come out the same from
//! one build to the next. `std`'s `DefaultHasher` promises nothing of the kind —
//! a toolchain that changed it would make every settled target look changed and
//! offer all of them again — so this is SHA-256, cut to its first eight bytes.

use sha2::{Digest, Sha256};

/// Accumulates the parts of a target and answers with their fingerprint.
pub struct Fingerprint(Sha256);

impl Fingerprint {
    pub fn new() -> Self {
        Self(Sha256::new())
    }

    /// Add one part. Each part is length-prefixed, so `("ab", "c")` and
    /// `("a", "bc")` fingerprint differently.
    pub fn add(&mut self, part: &str) -> &mut Self {
        self.0.update((part.len() as u64).to_le_bytes());
        self.0.update(part.as_bytes());
        self
    }

    /// Add a part that may be absent; absent is distinct from empty.
    pub fn add_opt(&mut self, part: Option<&str>) -> &mut Self {
        match part {
            Some(p) => {
                self.0.update([1u8]);
                self.add(p)
            }
            None => {
                self.0.update([0u8]);
                self
            }
        }
    }

    pub fn finish(&self) -> u64 {
        let digest = self.0.clone().finalize();
        let mut first = [0u8; 8];
        first.copy_from_slice(&digest[..8]);
        u64::from_le_bytes(first)
    }
}

impl Default for Fingerprint {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The value is fixed: SHA-256 of the length-prefixed bytes, first eight
    /// bytes little-endian. A change here re-offers every settled target.
    #[test]
    fn a_fingerprint_is_stable_across_builds() {
        assert_eq!(Fingerprint::new().finish(), 0x141c_fc98_42c4_b0e3);
        assert_eq!(
            Fingerprint::new().add("layers/eras/a.md").finish(),
            Fingerprint::new().add("layers/eras/a.md").finish()
        );
    }

    #[test]
    fn parts_are_delimited_and_absence_is_not_emptiness() {
        let ab_c = Fingerprint::new().add("ab").add("c").finish();
        let a_bc = Fingerprint::new().add("a").add("bc").finish();
        assert_ne!(ab_c, a_bc);
        let none = Fingerprint::new().add_opt(None).finish();
        let empty = Fingerprint::new().add_opt(Some("")).finish();
        assert_ne!(none, empty);
    }
}
