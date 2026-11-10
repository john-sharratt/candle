//! Sealing a calibration: its cases prefilled in one turn group of a
//! conversation of the hidden calibration layer, then archived and distilled
//! provenance-only, so its K/V is dropped and its signatures stay.
//!
//! A drafted calibration ([`super::run`]) and a use-exemplar
//! (`crate::guidance`) are sealed the same way; only their cases and their
//! naming metadata differ.

use std::collections::BTreeMap;

use candle_conversation::persistence::record::DistillMode;
use candle_conversation::projection::{DecodePriority, Reserved, SelectionState, TimelineId};
use candle_conversation::{ConversationEngine, SequenceConfig};

use super::run::{Frame, TAG};

/// The config a calibration is prefilled under: the frame's, lossless until
/// the seal, as the tools' calibrations are — the signature is captured from
/// every token's Q at the seal, and a quantized K/V would capture it from a
/// degraded one.
fn calibration_config(frame: &SequenceConfig) -> SequenceConfig {
    SequenceConfig {
        kv_lossless: true,
        ..frame.clone()
    }
}

/// Prefill `cases` — each a user half and the assistant half that answers it —
/// as a calibration named by `meta`, every case masked to itself and sealed one
/// turn apiece, then archive and distill it. Returns its timeline. A
/// calibration that fails to seal is tombstoned.
pub(crate) fn seal_calibration(
    engine: &ConversationEngine,
    frame: &Frame<'_>,
    meta: &BTreeMap<String, String>,
    cases: &[(String, String)],
) -> anyhow::Result<TimelineId> {
    let mut conv = engine
        // Background, as the ingest is: its prefill creeps beside a live
        // turn rather than competing with it (`scheduler::band`).
        .new_reserved_conversation(
            frame.system_prompt,
            Reserved::Calibration,
            DecodePriority::Low,
            calibration_config(frame.config),
        )
        .map_err(|e| anyhow::anyhow!("calibration conversation: {e}"))?;
    let timeline = conv.timeline_id();
    // Its K/V is dropped at the distill; written cold, it would be bytes the
    // next compaction deletes.
    engine.conversation().mark_timeline_transient(timeline);
    let sealed = conv
        .set_metadata_many(meta)
        .map_err(|e| anyhow::anyhow!("naming the calibration: {e}"))
        .and_then(|()| {
            let tagged: Vec<(String, String, Vec<String>)> = cases
                .iter()
                .map(|(user, answer)| (user.clone(), answer.clone(), vec![TAG.to_string()]))
                .collect();
            let (handle, _) = conv
                .submit_prefilled_turn_group(&tagged, SelectionState::new())
                .map_err(|e| anyhow::anyhow!("submit: {e}"))?;
            let resp = handle
                .wait_cancellable(crate::ingest_cancel::token())
                .map_err(|e| anyhow::anyhow!("prefill: {e}"))?;
            conv.finish_turn(handle, &resp)
                .map_err(|e| anyhow::anyhow!("seal: {e}"))?;
            Ok(())
        });
    if let Err(err) = sealed {
        let _ = engine.tombstone_timeline(timeline);
        return Err(err);
    }
    engine
        .set_conversation_archived(timeline, true)
        .map_err(|err| anyhow::anyhow!("archiving the calibration: {err}"))?;
    // Distilled whether or not any chunk reached disk: transient, none did,
    // and a peer holds the turns as still arriving until the marker says
    // their K/V was shed.
    if !engine.is_timeline_distilled(timeline) {
        engine
            .distill_timeline(timeline, DistillMode::ProvenanceOnly)
            .map_err(|err| anyhow::anyhow!("distilling the calibration: {err}"))?;
    }
    Ok(timeline)
}
