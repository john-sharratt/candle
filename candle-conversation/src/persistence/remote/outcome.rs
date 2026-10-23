//! What applying a remote record came to (`docs/replicated_substrate.md`
//! §6.5).

use super::super::appended::Appended;
use super::super::record::RecordType;
use crate::projection::{TimelineId, TurnIndex};

/// A remote record's check before it is applied
/// (`SubstratePersistence::check_remote`): passed, or why it is refused.
pub type Checked = Result<(), Refusal>;

/// Why a remote record was refused. A refused record is neither written nor
/// marked applied, so a catch-up fetches it again.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// It does not parse as a record.
    Malformed(String),
    /// Its payload does not hash to the `digest` its header names.
    Digest,
    /// Its payload does not match the header's CRC, or a `Chunk`'s golden.
    Checksum,
    /// It carries no identity, so it is not an event any member sent.
    Unidentified,
    /// It carries the provisional generation of a member cut off, which no
    /// member sends: its writer sends it once a rejoin has fixed its
    /// generation (§6.7).
    Detached,
    /// It is a type that never leaves the member that wrote it (§6.1).
    LocalOnly(RecordType),
    /// It is this member's own record, past every `oseq` this member issued:
    /// another member holds what this one lost.
    OwnOrigin,
}

/// What became of one remote record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RemoteOutcome {
    /// Its type is one this build does not decode: neither written nor
    /// applied, and recorded as deferred.
    Deferred,
    /// Already applied here.
    Duplicate,
    /// A tombstone here covers it: marked applied, not written.
    Dominated,
    /// The value it sets is held here under a greater stamp, so it would lose
    /// on arrival: marked applied, not written.
    Superseded,
    /// Written, and applied.
    Written(Appended),
    Refused(Refusal),
}

/// What a batch came to.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RemoteBatch {
    /// One outcome per record, in the batch's order.
    pub outcomes: Vec<RemoteOutcome>,
    /// The turns the batch completed, each placed at its position.
    pub adopted: Vec<(TimelineId, TurnIndex)>,
    /// A turn arrived ranked below a turn this member holds, so the indices of
    /// a live timeline changed: the member must reload (§6.6). Nothing is
    /// adopted from a batch that needs one; the reload restores it.
    pub reload: bool,
}
