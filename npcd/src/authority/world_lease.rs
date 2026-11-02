//! The world lease (`docs/replicated_substrate.md` §9.2, §9.6): `world/<wid>`,
//! held by Paxos among the npcd members. **A world stays where its lease is**
//! (`docs/static_placement.md` §6): the holder renews it every 10 s for as long
//! as it runs, and nothing — no rank, plan or rendezvous — takes it away. A
//! world nobody holds is asked for by its proposer, the ready member admitted
//! at the greatest power ([`proposer`]); it never takes it from a holder.

use std::time::Duration;

use cluster::consensus::rendezvous::Weighted;

/// The Paxos key of world `wid`'s lease.
pub fn lease_key(wid: &str) -> Vec<u8> {
    format!("world/{wid}").into_bytes()
}

/// The world a lease key names, if it is a world's.
pub fn world_of(key: &[u8]) -> Option<&str> {
    std::str::from_utf8(key).ok()?.strip_prefix("world/")
}

/// Who asks for a world nobody holds, among the `ready` members as they were
/// admitted: the greatest weight, then the lower node — so of BlackWorld (734)
/// and WhiteWorld (245), BlackWorld. A member of weight 0 takes no lease and
/// proposes for none.
pub fn proposer(ready: &[Weighted]) -> Option<u64> {
    ready
        .iter()
        .filter(|m| m.weight > 0.0)
        .max_by(|a, b| a.weight.total_cmp(&b.weight).then(b.node.cmp(&a.node)))
        .map(|m| m.node)
}

/// What to do about one world's lease this tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    /// Ask for it: this member is the proposer and does not hold it.
    Acquire,
    /// Renew it: held, and due.
    Renew,
    /// Nothing this tick.
    Wait,
}

/// The step for a world whose proposer is `proposer`, which this member, `me`,
/// holds or not, last asked for `since` ago. A holder renews whoever proposes:
/// the world stays where it is.
pub fn step(me: u64, proposer: Option<u64>, held: bool, since: Duration, every: Duration) -> Step {
    match (held, proposer == Some(me)) {
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

    /// **The proposer asks, and a holder renews whoever proposes**: a world
    /// stays where its lease is.
    #[test]
    fn the_proposer_asks_and_the_holder_keeps_it() {
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
        assert_eq!(
            step(1, Some(2), true, s(10), EVERY),
            Step::Renew,
            "another proposes: the holder keeps it"
        );
        assert_eq!(step(1, None, true, s(10), EVERY), Step::Renew);
    }

    /// **The proposer is the member admitted at the greatest power**, the lower
    /// node on a tie; a member that takes no lease proposes for none.
    #[test]
    fn the_proposer_is_the_strongest_ready_member() {
        let w = |node, weight| Weighted { node, weight };
        assert_eq!(
            proposer(&[w(0xb2e6, 245.0), w(0xdb1f, 734.0)]),
            Some(0xdb1f)
        );
        assert_eq!(
            proposer(&[w(9, 100.0), w(3, 100.0)]),
            Some(3),
            "the lower node"
        );
        assert_eq!(proposer(&[w(1, 0.0)]), None, "takes no lease");
        assert_eq!(proposer(&[]), None);
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
