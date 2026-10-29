//! This process's card id (`docs/cooperative_admission.md` §4.2): drawn at
//! random once, the first time it is asked for, and the same for every member
//! the process runs. zend's member and npcd's member in one motherd send it in
//! their `HELLO`s, so the gateway counts the work it places on either against
//! the one card they share.

use std::sync::OnceLock;

/// This process's card id: never zero, which names no card.
pub fn this() -> u64 {
    static CARD: OnceLock<u64> = OnceLock::new();
    *CARD.get_or_init(|| loop {
        let id: u64 = rand::random();
        if id != 0 {
            break id;
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One id for the life of the process, and never zero.
    #[test]
    fn a_process_has_one_card() {
        let first = this();
        assert_ne!(first, 0);
        assert_eq!(this(), first);
        let other = std::thread::spawn(this).join().unwrap();
        assert_eq!(other, first, "every thread of the process names one card");
    }
}
