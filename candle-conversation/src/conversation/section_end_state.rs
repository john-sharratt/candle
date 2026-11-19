//! The state a shared run's last section ended on, as its record keeps it
//! (`docs/shared_journal_sections.md` §4.1, §5).
//!
//! The record is a [`SnapshotPayload`] under the section's
//! `section_end_state_stream_id`, so it rides the snapshot single tail and
//! replicates with it. It belongs to no conversation and no turn, so those
//! fields are zero; [`SnapshotPayload::is_retired`]'s shape — no layers, no
//! aux — reads as nothing kept.

use crate::persistence::record::{SnapshotLayer, SnapshotPayload};
use crate::scheduler::exported_state::SharedState;
use candle_transformers::models::delta_net::ExportedLayerState;

/// `state` as the record keeps it.
pub(crate) fn to_payload(state: &SharedState) -> SnapshotPayload {
    SnapshotPayload {
        timeline_id: 0,
        turn_index: 0,
        schedule_hash: state.schedule_hash,
        layers: state
            .layers
            .iter()
            .cloned()
            .map(SnapshotLayer::from)
            .collect(),
        aux: state.aux.to_vec(),
    }
}

/// The state a record keeps; `None` for a retired one.
pub(crate) fn from_payload(payload: &SnapshotPayload) -> Option<SharedState> {
    if payload.is_retired() {
        return None;
    }
    let layers: Vec<ExportedLayerState> = payload
        .layers
        .iter()
        .map(|l| ExportedLayerState {
            layer_index: l.layer_index,
            n_v_heads: l.n_v_heads,
            d_v: l.d_v,
            d_k: l.d_k,
            state: l.state.clone(),
            conv_channels: l.conv_channels,
            conv_tail_cols: l.conv_tail_cols,
            conv_tail: l.conv_tail.clone(),
        })
        .collect();
    Some(SharedState {
        schedule_hash: payload.schedule_hash,
        layers: layers.into(),
        aux: payload.aux.clone().into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> SharedState {
        let layers: Vec<ExportedLayerState> = vec![ExportedLayerState {
            layer_index: 3,
            n_v_heads: 2,
            d_v: 1,
            d_k: 1,
            state: vec![1, 2, 3, 4, 5, 6, 7, 8],
            conv_channels: 1,
            conv_tail_cols: 2,
            conv_tail: vec![9, 10, 11, 12, 13, 14, 15, 16],
        }];
        SharedState {
            schedule_hash: 0xFEED,
            layers: layers.into(),
            aux: vec![0xA1, 0xA2].into(),
        }
    }

    /// The state survives the record byte for byte, through its encoding.
    #[test]
    fn a_kept_end_state_round_trips_through_its_encoding() {
        let payload = to_payload(&state());
        assert_eq!(
            (
                payload.timeline_id,
                payload.turn_index,
                payload.schedule_hash
            ),
            (0, 0, 0xFEED)
        );
        let decoded = SnapshotPayload::decode(&payload.encode()).unwrap();
        let back = from_payload(&decoded).expect("a full record");
        assert_eq!(back.schedule_hash, 0xFEED);
        assert_eq!(back.layers[0].layer_index, 3);
        assert_eq!(back.layers[0].state, vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(
            back.layers[0].conv_tail,
            vec![9, 10, 11, 12, 13, 14, 15, 16]
        );
        assert_eq!(&*back.aux, &[0xA1, 0xA2]);
    }

    #[test]
    fn a_retired_record_keeps_nothing() {
        assert!(from_payload(&SnapshotPayload::retired(0)).is_none());
    }
}
