//! Whether a model member steps aside for the machine's reservation.
//!
//! **It drains only while every app it serves has another member ready.** A
//! party is run by Zen — it opens the decks, sets the house, refuses to leave
//! while a deck plays — so a reservation that took the only member serving an
//! app down would take down the thing that asked for it. While another member
//! is ready for every app, this one drains: the gateways route it nothing new,
//! its claims and leases go to the others, and it stays connected, so it keeps
//! seeing who else is ready. The moment one app has nobody else, it undrains
//! and serves it, reservation or not.
//!
//! Only a drain the reservation made is undone by it; one a person asked for
//! stays.

/// What the member does now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Drain every app: the others cover them all.
    Drain,
    /// Undo the reservation's drain: it was released, or an app lost its
    /// last other member.
    Undrain,
    /// Nothing changes.
    Stay,
}

/// The step for a member whose machine is `reserved`, with `peers_ready` the
/// count of other ready members for each app it serves, having drained for
/// the reservation already when `drained`.
pub fn step(reserved: bool, peers_ready: &[usize], drained: bool) -> Step {
    let covered = !peers_ready.is_empty() && peers_ready.iter().all(|n| *n > 0);
    match (reserved && covered, drained) {
        (true, false) => Step::Drain,
        (false, true) => Step::Undrain,
        _ => Step::Stay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **It steps aside only while every app has someone else ready**, and
    /// steps back the moment one does not or the reservation ends.
    #[test]
    fn a_member_drains_only_while_every_app_is_covered() {
        assert_eq!(step(true, &[1, 1], false), Step::Drain);
        assert_eq!(step(true, &[1, 1], true), Step::Stay, "drained already");
        assert_eq!(
            step(true, &[1, 0], false),
            Step::Stay,
            "npcd has nobody else: it keeps serving"
        );
        assert_eq!(
            step(true, &[1, 0], true),
            Step::Undrain,
            "an app lost its other member"
        );
        assert_eq!(step(false, &[1, 1], true), Step::Undrain, "released");
        assert_eq!(step(false, &[1, 1], false), Step::Stay);
        assert_eq!(step(true, &[], false), Step::Stay, "no apps joined");
    }
}
