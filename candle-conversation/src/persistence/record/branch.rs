//! The payload of a prompt branch's recurrent checkpoint.
//!
//! A checkpoint holds the recurrent state after a whole system prompt, for one
//! selector assignment, so a new conversation can begin its first turn with the
//! state the prompt would have produced. It shares the snapshot record type with a
//! conversation's snapshot, and the stream id is what tells them apart. A leading
//! magic makes a misrouted read fail rather than decode as a plausible state. The
//! payload carries model-opaque aux state as well.

use super::*;

/// Binary payload for a [`RecordType::Snapshot`] record written under a
/// **branch checkpoint** stream — the recurrent state after a whole system
/// prompt, for one selector assignment.
///
/// A new conversation does not prefill its system prompt: it Arc-injects sealed
/// section K/V, so the wave never sees those tokens and the recurrent state
/// would enter the first user turn at zero while the attention layers hold the
/// entire prompt. This record is what removes that asymmetry, and it is the
/// only place a conversation needs a state it did not compute
/// (`docs/deltanet_state_persistence.md` §4.6).
///
/// **Two payload shapes share `RecordType::Snapshot`, distinguished by stream
/// id.** That is not an overload: every piece of the single-tail machinery —
/// the accounting supersede key, the recovery walk's location map, the
/// compactor's carry-forward — keys on `header.stream_id` and never decodes the
/// payload. The stream id *is* the identity of what the state belongs to, and a
/// reader computes it before asking, so it always knows which shape it will get.
/// A conversation's snapshot is keyed by `snapshot_stream_id(timeline)`, a
/// branch's by [`branch_checkpoint_stream_id`](super::content_hash::branch_checkpoint_stream_id).
///
/// `prefix_hash` is stored as well as keyed on, so a read can confirm it got
/// the branch it asked for rather than trusting a 64-bit stream id not to
/// collide. `schedule_hash` fingerprints the model's layer schedule + DeltaNet
/// dims exactly as it does for a conversation snapshot: a checkpoint computed
/// under one geometry must never be scattered into another.
#[derive(Clone, Debug, PartialEq)]
pub struct BranchCheckpointPayload {
    /// The branch's cumulative content prefix — its identity.
    pub prefix_hash: ContentHash,
    pub schedule_hash: u64,
    pub layers: Vec<SnapshotLayer>,
    /// Model-opaque state, exactly as [`SnapshotPayload::aux`].
    ///
    /// A checkpoint exists so a conversation enters its first user turn holding
    /// the state the system prompt would have produced. That argument does not
    /// single out the delta-rule matrices: a model whose *whole* recurrence is
    /// in the aux blob and whose checkpoint carried only `layers` would restore
    /// an empty index over a prompt's worth of attention K/V — the same
    /// asymmetry this record exists to remove, one class of state further in.
    pub aux: Vec<u8>,
}

/// Leading magic, so the two payloads sharing [`RecordType::Snapshot`] can
/// never be mistaken for one another on the wire.
///
/// Without it they are not merely similar, they are **compatible**: a
/// conversation snapshot decodes as a branch checkpoint without error, because
/// `(version, timeline, turn, schedule, n_layers)` and
/// `(version, prefix_lo, prefix_hi, schedule, n_layers)` are the same widths in
/// the same order, so every field lands somewhere plausible. A misrouted read
/// would then produce a *state*, not an error — the failure mode this whole
/// area exists to remove. `SnapshotPayload`'s own version field is `1`, which
/// this magic cannot collide with, so the refusal runs in both directions.
pub(super) const BRANCH_CHECKPOINT_MAGIC: &[u8; 4] = b"BRCK";

/// Wire version of [`BranchCheckpointPayload`]. Version 2 appends `aux`.
pub(super) const BRANCH_CHECKPOINT_VERSION: u32 = 2;

impl BranchCheckpointPayload {
    /// The payload as the slices it is written from — see
    /// [`SnapshotPayload::parts`].
    pub fn parts(&self) -> PayloadParts<'_> {
        let mut p = PayloadParts::default();
        p.raw(BRANCH_CHECKPOINT_MAGIC);
        p.u32(BRANCH_CHECKPOINT_VERSION);
        p.u64(self.prefix_hash.lo);
        p.u64(self.prefix_hash.hi);
        p.u64(self.schedule_hash);
        p.u32(self.layers.len() as u32);
        for l in &self.layers {
            snapshot_layer_parts(&mut p, l);
        }
        p.blob(&self.aux);
        p
    }

    pub fn encode(&self) -> Vec<u8> {
        self.parts().concat()
    }

    pub fn decode(buf: &[u8]) -> Result<Self> {
        let mut r = ByteReader::new(buf);
        for (i, &want) in BRANCH_CHECKPOINT_MAGIC.iter().enumerate() {
            let got = r.get_u8()?;
            if got != want {
                return Err(PersistenceError::Corrupt(format!(
                    "not a branch checkpoint: magic byte {i} is {got:#04x}, expected \
                     {want:#04x}. A conversation snapshot has the same field widths \
                     in the same order and would otherwise decode as a plausible \
                     branch state."
                )));
            }
        }
        let version = r.get_u32()?;
        if version != BRANCH_CHECKPOINT_VERSION {
            return Err(PersistenceError::Corrupt(format!(
                "branch checkpoint payload version {version} unknown (this build \
                 reads {BRANCH_CHECKPOINT_VERSION})"
            )));
        }
        let lo = r.get_u64()?;
        let hi = r.get_u64()?;
        let schedule_hash = r.get_u64()?;
        let n_layers = r.get_u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(decode_snapshot_layer(&mut r)?);
        }
        let aux = r.get_blob()?.to_vec();
        Ok(Self {
            prefix_hash: ContentHash { lo, hi },
            schedule_hash,
            layers,
            aux,
        })
    }
}
