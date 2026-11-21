//! The payload of a conversation's recurrent-state snapshot, taken at a turn's seal.
//!
//! The snapshot carries each recurrent layer's matrices and conv tails, a
//! model-opaque aux blob that only the writing model reads, and a schedule hash
//! that refuses a snapshot taken under a different layer layout. The version is
//! bumped on any layout change, and an older version is refused rather than read as
//! a plausible state. The state is written from borrowed parts, so a seal does not
//! copy it first.

use super::*;

/// One recurrent layer's state inside a [`SnapshotPayload`].
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotLayer {
    /// Trunk layer index this state belongs to.
    pub layer_index: u32,
    /// `[n_v_heads × d_v × d_k]` F32 delta-rule matrices, LE bytes.
    pub n_v_heads: u32,
    pub d_v: u32,
    pub d_k: u32,
    pub state: Vec<u8>,
    /// `[conv_channels × conv_tail_cols]` F32 conv tail, LE bytes.
    pub conv_channels: u32,
    pub conv_tail_cols: u32,
    pub conv_tail: Vec<u8>,
}

impl From<ExportedLayerState> for SnapshotLayer {
    /// The model's export rows are field-for-field this record's layer rows —
    /// `candle-conversation` depends on `candle-transformers` and not the
    /// reverse, so the model hands back its own type and the record shape is
    /// built here.
    fn from(l: ExportedLayerState) -> Self {
        Self {
            layer_index: l.layer_index,
            n_v_heads: l.n_v_heads,
            d_v: l.d_v,
            d_k: l.d_k,
            state: l.state,
            conv_channels: l.conv_channels,
            conv_tail_cols: l.conv_tail_cols,
            conv_tail: l.conv_tail,
        }
    }
}

impl From<SnapshotLayer> for ExportedLayerState {
    /// The record's rows handed to the model as they are: a restore moves the
    /// decoded rows into the snapshot it stages rather than copying them.
    fn from(mut l: SnapshotLayer) -> Self {
        Self {
            layer_index: l.layer_index,
            n_v_heads: l.n_v_heads,
            d_v: l.d_v,
            d_k: l.d_k,
            state: std::mem::take(&mut l.state),
            conv_channels: l.conv_channels,
            conv_tail_cols: l.conv_tail_cols,
            conv_tail: std::mem::take(&mut l.conv_tail),
        }
    }
}

impl Drop for SnapshotLayer {
    /// A layer's rows go back to the host pool its decode or its seal's export
    /// took them from, so the next one fills them rather than allocating.
    fn drop(&mut self) {
        host_rows::give(std::mem::take(&mut self.state));
        host_rows::give(std::mem::take(&mut self.conv_tail));
    }
}

/// Binary payload for a [`RecordType::Snapshot`] record — a conversation's
/// full recurrent state at the seal of `turn_index`.
///
/// `schedule_hash` fingerprints the model's layer schedule + DeltaNet dims;
/// restore refuses a snapshot whose hash disagrees with the loaded model
/// (a resumed conversation must recompute instead of scattering a foreign
/// layout into the state arena). `turn_index` binds the snapshot to the turn
/// whose seal produced it: reload discards a snapshot newer than the last
/// recovered turn (a torn shutdown between the turn's records and this one)
/// and falls back to recompute.
#[derive(Clone, Debug, PartialEq)]
pub struct SnapshotPayload {
    pub timeline_id: u64,
    pub turn_index: u32,
    pub schedule_hash: u64,
    pub layers: Vec<SnapshotLayer>,
    /// Model-opaque state this record carries alongside `layers`.
    ///
    /// `layers` is one shape — a per-layer delta-rule matrix plus a conv tail —
    /// and it is the right shape for the state that owns this record. It is not
    /// the only recurrence a model can carry: Flash-Next also holds a PLE cache
    /// and a QSA index, neither of which is a delta-rule matrix and neither of
    /// which fits a field of `SnapshotLayer` without inventing one per model.
    ///
    /// So this is bytes, and only the model that wrote them reads them
    /// ([`export_aux_state`](candle_transformers::models::batched_inference::ManagedBatchedModel::export_aux_state)).
    /// Persistence carries the blob, checksums it with the rest of the record,
    /// and never decodes it — which is what keeps one seal path serving every
    /// architecture instead of growing a branch per model. Empty for a model
    /// with no such state, which costs a length word.
    pub aux: Vec<u8>,
}

/// Wire version of [`SnapshotPayload`]; bump on layout change.
///
/// Version 2 appends the model-opaque `aux` blob. A version-1 record decodes to
/// a *plausible* state under this layout — the trailing length word would be
/// read from whatever followed the layers — so the mismatch is refused rather
/// than tolerated, and a substrate written by an older build recomputes.
pub(super) const SNAPSHOT_PAYLOAD_VERSION: u32 = 2;

impl SnapshotPayload {
    /// The tail of a timeline that keeps no recurrent state: no layers, no aux.
    /// Written to supersede a full snapshot nothing will read again
    /// (`Conversation::mark_timeline_unresumed`); read back as no snapshot.
    pub fn retired(timeline_id: u64) -> Self {
        Self {
            timeline_id,
            turn_index: 0,
            schedule_hash: 0,
            layers: Vec::new(),
            aux: Vec::new(),
        }
    }

    /// Whether this is a [`Self::retired`] tail. No model writes a snapshot
    /// with neither layers nor aux — a model with no recurrent state writes
    /// none at all — so the shape is unambiguous.
    pub fn is_retired(&self) -> bool {
        self.layers.is_empty() && self.aux.is_empty()
    }

    /// The exact length [`Self::encode`] produces, counted from the fields
    /// without encoding anything — what the writer's byte cap charges a queued
    /// snapshot, and what the seal's snapshot-size counter adds.
    pub fn encoded_len(&self) -> usize {
        // version, timeline_id, turn_index, schedule_hash, layer count …
        let header = 4 + 8 + 4 + 8 + 4;
        // … each layer: index, dtype tag, three dims, state blob, two conv dims,
        // conv-tail blob …
        let layers: usize = self
            .layers
            .iter()
            .map(|l| 4 + 1 + 4 + 4 + 4 + (4 + l.state.len()) + 4 + 4 + (4 + l.conv_tail.len()))
            .sum();
        // … and the aux blob.
        header + layers + 4 + self.aux.len()
    }

    /// The payload as the slices it is written from: the state blobs are
    /// borrowed, so a seal writes them to the file without first copying them
    /// into one buffer. See [`PayloadParts`].
    pub fn parts(&self) -> PayloadParts<'_> {
        let mut p = PayloadParts::default();
        p.u32(SNAPSHOT_PAYLOAD_VERSION);
        p.u64(self.timeline_id);
        p.u32(self.turn_index);
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
        let version = r.get_u32()?;
        if version != SNAPSHOT_PAYLOAD_VERSION {
            return Err(PersistenceError::Corrupt(format!(
                "snapshot payload version {version} unknown (this build reads \
                 {SNAPSHOT_PAYLOAD_VERSION})"
            )));
        }
        let timeline_id = r.get_u64()?;
        let turn_index = r.get_u32()?;
        let schedule_hash = r.get_u64()?;
        let n_layers = r.get_u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            layers.push(decode_snapshot_layer(&mut r)?);
        }
        let aux = r.get_blob()?.to_vec();
        Ok(Self {
            timeline_id,
            turn_index,
            schedule_hash,
            layers,
            aux,
        })
    }
}

/// One layer's state, in the shared on-wire form both recurrent payloads use.
pub(super) fn snapshot_layer_parts<'a>(p: &mut PayloadParts<'a>, l: &'a SnapshotLayer) {
    p.u32(l.layer_index);
    p.u8(0); // dtype tag: 0 = F32 (the only state dtype)
    p.u32(l.n_v_heads);
    p.u32(l.d_v);
    p.u32(l.d_k);
    p.blob(&l.state);
    p.u32(l.conv_channels);
    p.u32(l.conv_tail_cols);
    p.blob(&l.conv_tail);
}

pub(super) fn decode_snapshot_layer(r: &mut ByteReader<'_>) -> Result<SnapshotLayer> {
    let layer_index = r.get_u32()?;
    let dtype = r.get_u8()?;
    if dtype != 0 {
        return Err(PersistenceError::Corrupt(format!(
            "snapshot layer {layer_index}: unknown state dtype tag {dtype}"
        )));
    }
    let n_v_heads = r.get_u32()?;
    let d_v = r.get_u32()?;
    let d_k = r.get_u32()?;
    // Into a kept row: a restore's rows are the size every snapshot of the model
    // has, and the last snapshot dropped left its rows in the pool.
    let state = host_rows::row_from(r.get_blob()?);
    let expect = n_v_heads as usize * d_v as usize * d_k as usize * 4;
    if state.len() != expect {
        return Err(PersistenceError::Corrupt(format!(
            "snapshot layer {layer_index}: state blob {} bytes, dims say {expect}",
            state.len()
        )));
    }
    let conv_channels = r.get_u32()?;
    let conv_tail_cols = r.get_u32()?;
    let conv_tail = host_rows::row_from(r.get_blob()?);
    let expect_tail = conv_channels as usize * conv_tail_cols as usize * 4;
    if conv_tail.len() != expect_tail {
        return Err(PersistenceError::Corrupt(format!(
            "snapshot layer {layer_index}: conv tail {} bytes, dims say {expect_tail}",
            conv_tail.len()
        )));
    }
    Ok(SnapshotLayer {
        layer_index,
        n_v_heads,
        d_v,
        d_k,
        state,
        conv_channels,
        conv_tail_cols,
        conv_tail,
    })
}
