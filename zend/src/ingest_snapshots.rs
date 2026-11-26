//! An analysis layer's units keep no recurrent snapshot.
//!
//! A unit of a layer built on another is read once and never continued: a
//! dialogue borrows its K/V through projection, and a changed subject is read
//! again as a new conversation. The snapshot each of its seals wrote — one GDN
//! store, 113 MB on this model, fixed whatever the turn's length — is never
//! read back, and the substrate would keep one live tail per conversation for
//! good. Measured on BlackWorld, 2026-10-19: the substrate grew from 332 GB to
//! 526 GB in twelve hours of ingest and load, with only 6.7% of it dead.

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

/// `timeline` has finished reading its unit: its tail is retired and no later
/// seal writes another.
pub fn finished(engine: &ConversationEngine, timeline: TimelineId) {
    engine.mark_timeline_unresumed(timeline);
}
