//! Who wrote a record, in what order, and where it ranks against a rival.
//!
//! Every record a substrate writes carries the identity of the substrate that
//! wrote it (`origin`) and its place in that substrate's own sequence (`oseq`),
//! so the record means the same thing on every machine that holds a copy
//! (`docs/replicated_substrate.md` §6.2). The rest of the identity — `gen`,
//! `epoch` and `hlc` — is the **version stamp** that orders two records
//! competing for one key, the same way on every machine.

use std::cmp::Ordering;

/// The provisional generation of a record written while its substrate was cut
/// off from every cluster: above every generation the group can mint, until a
/// rejoin fixes it (§6.7).
pub const GEN_DETACHED: u64 = u64::MAX;

/// A record's identity, as its header carries it.
///
/// All zero for a record written before identities existed. Such a record
/// belongs to no origin and ranks below every stamped record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct RecordIdentity {
    /// The node of the substrate that wrote it (§3.7).
    pub origin: u64,
    /// That origin's count of the records it has written, from 1.
    pub oseq: u64,
    /// The generation the writer stood in: its current one, or
    /// [`GEN_DETACHED`] while cut off.
    pub gen: u64,
    /// The claim or lease epoch it was written under; zero when nothing
    /// governs it.
    pub epoch: u64,
    /// The hybrid logical clock reading it was written at ([`super::Hlc`]).
    pub hlc: u64,
}

impl RecordIdentity {
    /// The identity of a record written before identities existed, and of a
    /// record no origin writes (the header index).
    pub const NONE: RecordIdentity = RecordIdentity {
        origin: 0,
        oseq: 0,
        gen: 0,
        epoch: 0,
        hlc: 0,
    };

    /// Whether the record carries no identity.
    pub fn is_none(&self) -> bool {
        self.origin == 0 && self.oseq == 0
    }

    /// The version stamp two competing records are ordered by.
    pub fn stamp(&self) -> VersionStamp {
        VersionStamp {
            gen: self.gen,
            epoch: self.epoch,
            hlc: self.hlc,
            origin: self.origin,
        }
    }
}

/// The order of two records competing for one key: `(gen, epoch, hlc,
/// origin)`, compared in that order (§6.2). Every machine computes it the same
/// way, so the greater stamp wins everywhere whichever copy arrived first.
///
/// `hlc` is strictly increasing per origin, so two stamped records of one
/// origin never tie, and `origin` breaks a tie between origins.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct VersionStamp {
    pub gen: u64,
    pub epoch: u64,
    pub hlc: u64,
    pub origin: u64,
}

impl Ord for VersionStamp {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.gen, self.epoch, self.hlc, self.origin).cmp(&(
            other.gen,
            other.epoch,
            other.hlc,
            other.origin,
        ))
    }
}

impl PartialOrd for VersionStamp {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(gen: u64, epoch: u64, hlc: u64, origin: u64) -> RecordIdentity {
        RecordIdentity {
            origin,
            oseq: 1,
            gen,
            epoch,
            hlc,
        }
    }

    /// The fields rank in the stated order: a generation outranks any epoch,
    /// an epoch any clock reading, a clock reading any origin.
    #[test]
    fn the_stamp_ranks_gen_then_epoch_then_hlc_then_origin() {
        assert!(id(2, 0, 0, 0).stamp() > id(1, 9, 9, 9).stamp());
        assert!(id(1, 2, 0, 0).stamp() > id(1, 1, 9, 9).stamp());
        assert!(id(1, 1, 2, 0).stamp() > id(1, 1, 1, 9).stamp());
        assert!(id(1, 1, 1, 2).stamp() > id(1, 1, 1, 1).stamp());
        assert_eq!(id(1, 1, 1, 1).stamp(), id(1, 1, 1, 1).stamp());
    }

    /// A detached record sorts after everything a joined writer can stamp.
    #[test]
    fn a_detached_generation_outranks_every_minted_one() {
        assert!(id(GEN_DETACHED, 0, 0, 1).stamp() > id(u64::MAX - 1, 9, 9, 9).stamp());
    }

    /// An unstamped record ranks below every stamped one.
    #[test]
    fn no_identity_ranks_lowest() {
        assert!(RecordIdentity::NONE.is_none());
        assert!(id(0, 0, 1, 1).stamp() > RecordIdentity::NONE.stamp());
    }
}
