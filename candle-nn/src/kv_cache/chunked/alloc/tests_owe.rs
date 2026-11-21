//! The arithmetic of how many arenas a refused pass is owed.
//!
//! A class short of slots is owed the arenas that cover its shortfall, once.
//! Owing again adds only what is not yet owed, a class with room is owed
//! nothing, and no demand is owed past the cap.

#[cfg(feature = "cuda")]
use super::*;

#[cfg(all(test, feature = "cuda"))]
pub(super) mod owe_tests {
    use super::{more_arenas_owed, MAX_OWED_PER_CLASS};

    /// A class short of slots is owed the arenas that cover the shortfall,
    /// once: owing again adds only what is not owed yet, a class with room is
    /// owed nothing, and no demand owes past the cap.
    #[test]
    fn a_class_is_owed_the_arenas_its_shortfall_needs() {
        // 300 slots wanted, 20 free, 16 a region: 280 short is 18 arenas.
        assert_eq!(more_arenas_owed(300, 20, 16, 0), 18);
        assert_eq!(more_arenas_owed(300, 20, 16, 18), 0, "already owed");
        assert_eq!(more_arenas_owed(300, 20, 16, 5), 13, "only the rest");
        assert_eq!(more_arenas_owed(10, 20, 16, 0), 0, "room enough");
        assert_eq!(more_arenas_owed(1 << 30, 0, 16, 0), MAX_OWED_PER_CLASS);
        assert_eq!(
            more_arenas_owed(5, 0, 0, 0),
            5,
            "a zero capacity reads as one"
        );
    }
}
