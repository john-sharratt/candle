//! The world lease (`docs/replicated_substrate.md` §9.2, §9.6): `world/<wid>`,
//! held by Paxos among the npcd members. Rendezvous over the members names who
//! *proposes* for it; it never takes it from a holder. The holder renews it
//! every 10 s while rendezvous still names it, and hands it over — releasing it
//! with its own `oseq` — when rendezvous names another, which then takes it.

use std::time::Duration;

/// The Paxos key of world `wid`'s lease.
pub fn lease_key(wid: &str) -> Vec<u8> {
    format!("world/{wid}").into_bytes()
}

/// The world a lease key names, if it is a world's.
pub fn world_of(key: &[u8]) -> Option<&str> {
    std::str::from_utf8(key).ok()?.strip_prefix("world/")
}

/// What to do about one world's lease this tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Ask for it: rendezvous names this member and it does not hold it.
    Acquire,
    /// Renew it: held, still named, and due.
    Renew,
    /// Hand it over: held, and rendezvous names another member now.
    HandOver,
    /// Nothing this tick.
    Wait,
}

/// The step for a world whose proposer rendezvous names as `proposer`, which
/// this member, `me`, holds or not, last asked for `since` ago.
pub fn step(me: u64, proposer: Option<u64>, held: bool, since: Duration, every: Duration) -> Step {
    let named = proposer == Some(me);
    match (held, named) {
        (true, false) if proposer.is_some() => Step::HandOver,
        (true, _) if since >= every => Step::Renew,
        (true, _) => Step::Wait,
        (false, true) if since >= ASK_AGAIN => Step::Acquire,
        _ => Step::Wait,
    }
}

/// How soon a proposer that did not get the lease asks again: about as long as
/// a round, so a release that has not reached every acceptor yet is waited out
/// without hammering them.
pub const ASK_AGAIN: Duration = Duration::from_secs(1);

/// The epoch a member gaining a world writes under: its clock, or above every
/// epoch the world's records carry if its clock is behind them (§9.2).
pub fn epoch_for(now_ms: u64, highest_seen: u64) -> u64 {
    now_ms.max(highest_seen + 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EVERY: Duration = Duration::from_secs(10);

    #[test]
    fn the_proposer_asks_the_holder_renews_and_hands_over() {
        let s = Duration::from_secs;
        assert_eq!(step(1, Some(1), false, s(5), EVERY), Step::Acquire);
        assert_eq!(
            step(1, Some(1), false, Duration::ZERO, EVERY),
            Step::Wait,
            "just asked"
        );
        assert_eq!(
            step(1, Some(2), false, s(60), EVERY),
            Step::Wait,
            "not its to ask"
        );
        assert_eq!(step(1, Some(1), true, s(3), EVERY), Step::Wait);
        assert_eq!(step(1, Some(1), true, s(10), EVERY), Step::Renew);
        assert_eq!(step(1, Some(2), true, s(3), EVERY), Step::HandOver);
        assert_eq!(
            step(1, None, true, s(10), EVERY),
            Step::Renew,
            "nobody else to hand to"
        );
    }

    #[test]
    fn a_world_lease_is_keyed_by_its_world() {
        assert_eq!(lease_key("vault"), b"world/vault".to_vec());
        assert_eq!(world_of(b"world/vault"), Some("vault"));
        assert_eq!(world_of(b"gen/3"), None);
    }

    /// The epoch rises above everything the world was written under.
    #[test]
    fn a_new_holders_epoch_is_above_every_one_seen() {
        assert_eq!(epoch_for(5_000, 1_000), 5_000);
        assert_eq!(epoch_for(5_000, 9_000), 9_001);
    }
}
